# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.models.video_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import str2bool, interpolate_log, create_folder, generate_log_json, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.png_reader import PNGReader
from tqdm import tqdm
from pytorch_msssim import ms_ssim

# 用于解析命令行参数
def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Example testing script")

    # 边缘增强强度参数，范围0.0-1.0
    parser.add_argument('--edge_alpha', type=float, help='Edge enhancement strength (0.0-1.0)')
    # I帧模型路径，字符串类型
    parser.add_argument('--i_frame_model_path', type=str)
    # I帧量化参数，可接受多个浮点数
    parser.add_argument('--i_frame_q_scales', type=float, nargs="+")
    # 是否强制使用I帧模式，布尔值，默认False
    parser.add_argument("--force_intra", type=str2bool, nargs='?', const=True, default=False)
    # 强制指定测试帧数，整数，默认-1
    parser.add_argument("--force_frame_num", type=int, default=-1)
    # 强制指定I帧间隔，整数，默认-1
    parser.add_argument("--force_intra_period", type=int, default=-1)
    # P帧模型路径，字符串类型
    parser.add_argument('--model_path',  type=str)
    # P帧Y分量量化参数，可接受多个浮点数
    parser.add_argument('--p_frame_y_q_scales', type=float, nargs="+")
    # P帧运动矢量量化参数，可接受多个浮点数
    parser.add_argument('--p_frame_mv_y_q_scales', type=float, nargs="+")
    # 码率点数，整数，默认4，要测试多少种不同的码率
    parser.add_argument('--rate_num', type=int, default=4)
    # 测试配置文件路径，字符串类型，必需参数
    parser.add_argument('--test_config', type=str, required=True)
    # 强制指定根路径，字符串类型，可选
    parser.add_argument('--force_root_path', type=str, default=None, required=False)
    # 工作进程数，整数，默认1
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    # 是否使用CUDA，布尔值，默认False
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    # 指定使用的CUDA设备，字符串类型
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    # 是否写入码流文件，布尔值，默认False
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    # 码流文件保存路径，字符串类型，默认"out_bin"
    parser.add_argument('--stream_path', type=str, default="out_bin")
    # 是否保存解码后的帧，布尔值，默认False
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    # 解码帧保存路径，字符串类型，默认'decoded_frames'
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    # 输出路径，字符串类型，必需参数
    parser.add_argument('--output_path', type=str, required=True)
    # 日志详细程度，整数，默认0
    parser.add_argument('--verbose', type=int, default=0)

    # 解析命令行参数
    args = parser.parse_args()
    return args


def read_image_to_torch(path, args):
    input_image = Image.open(path).convert('RGB')

    # 新增边缘增强处理（使用PyTorch原生实现）
    img_np = np.array(input_image).astype(np.float32)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # [C, H, W]
    
    # 定义Sobel算子
    sobel_x = torch.tensor([[[1, 0, -1], 
                           [2, 0, -2], 
                           [1, 0, -1]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[1, 2, 1], 
                           [0, 0, 0], 
                           [-1, -2, -1]]], dtype=torch.float32)
    
    # 计算边缘强度
    edge_x = F.conv2d(img_tensor.unsqueeze(0), sobel_x.unsqueeze(1), padding=1)
    edge_y = F.conv2d(img_tensor.unsqueeze(0), sobel_y.unsqueeze(1), padding=1)
    edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
    
    # 原图与边缘信息融合
    enhanced_img = torch.clamp(img_tensor + args['edge_alpha'] * edge_magnitude.squeeze(0), 0, 1)

    # 保持原有处理流程
    enhanced_img = enhanced_img.permute(1, 2, 0).numpy()  # 转回HWC格式
    input_image = Image.fromarray((enhanced_img * 255).astype(np.uint8))
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0)/255
    return input_image

def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()

def sobel_edge_enhancement(x, edge_alpha=0.3):
    """
    对输入的张量应用 Sobel 算子进行边缘增强。

    :param x: 输入的张量，形状为 (batch_size, channels, height, width)
    :param edge_alpha: 边缘增强的强度系数，默认为 0.3
    :return: 边缘增强后的张量
    """

    # repeat：将卷积核在第0维上重复x.shape[1]次，后两个维度不变，以确保每个通道都有一个卷积核。-> [channels, 3, 3]
    # unsqueeze：卷积核形状应为：[out_channels, in_channels/groups, kH, kW]，因为是逐通道卷积，所以in_channels/groups=1。-> [channels, 1, 3, 3]
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=x.device) # 形状为 [3, 3]
    sobel_x = sobel_x.repeat(x.shape[1], 1, 1).unsqueeze(1)

    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=x.device)
    sobel_y = sobel_y.repeat(x.shape[1], 1, 1).unsqueeze(1)  # 扩展为 [channels, 1, 3, 3] 格式
    
    # 执行卷积操作，设置 groups=x.shape[1] 进行分组卷积
    edge_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
    edge_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
    
    # 计算边缘强度
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    #todo 将边缘强度保存为图像
    # # 取批次中的第一张图，将多个通道的边缘强度取平均
    # edge_img = edge_magnitude[0].mean(dim=0).cpu().detach().numpy()
    # # 归一化到 0-255 范围
    # edge_img = (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min()) * 255
    # edge_img = edge_img.astype(np.uint8)
    # # 创建 PIL 图像并保存
    # img = Image.fromarray(edge_img)
    # img.save('edge.png')

    # 边缘增强
    enhanced_x = torch.clamp(x + edge_alpha * edge_magnitude, 0, 1)
    return enhanced_x

# 视频编码+视频解码
def run_test(video_net, i_frame_net, args, device):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    write_stream = 'write_stream' in args and args['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args and args['save_decoded_frame']
    verbose = args['verbose'] if 'verbose' in args else 0

    if args['src_type'] == 'png':
        #* 存放图片的文件夹，图片的宽，高
        src_reader = PNGReader(args['img_path'], args['src_width'], args['src_height'])

    frame_types = []
    psnrs = []
    msssims = []
    bits = []
    frame_pixel_num = 0

    start_time = time.time()
    p_frame_number = 0
    overall_p_decoding_time = 0
    with torch.no_grad():
        for frame_idx in range(frame_num):
            frame_start_time = time.time()

            #* 返回一张RGB图像（numpy数组，[C, H, W]）
            rgb = src_reader.read_one_frame(src_format="rgb") 

            #* 将numpy数组转化为PyTorch张量，在第0维增加batch维度，即(batch, C, H, W)
            #* batch=1，表示张量中只包含一张图像
            x = np_image_to_tensor(rgb) 

            x = x.to(device) #* 将张量移动到GPU上

            # todo 添加边缘增强
            if args['edge_alpha'] is not None:
                x = sobel_edge_enhancement(x, args['edge_alpha'])

            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
            x_padded = torch.nn.functional.pad(
                x,
                (padding_l, padding_r, padding_t, padding_b),
                mode="constant",
                value=0,
            )

            bin_path = os.path.join(args['bin_folder'], f"{frame_idx}.bin") \
                if write_stream else None

            # 处理I帧
            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_padded, args['i_frame_q_scale'], bin_path,
                                                   pic_height=pic_height, pic_width=pic_width)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"])
            # 处理P帧
            else:
                result = video_net.encode_decode(x_padded, dpb, bin_path,
                                                 pic_height=pic_height, pic_width=pic_width,
                                                 mv_y_q_scale=args['p_frame_mv_y_q_scale'],
                                                 y_q_scale=args['p_frame_y_q_scale'])
                dpb = result["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])
                p_frame_number += 1
                overall_p_decoding_time += result['decoding_time']

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            psnr = PSNR(x_hat, x)
            msssim = ms_ssim(x_hat, x, data_range=1).item()
            psnrs.append(psnr)
            msssims.append(msssim)
            frame_end_time = time.time()

            if verbose >= 2:
                print(f"frame {frame_idx}, {frame_end_time - frame_start_time:.3f} seconds,",
                      f"bits: {bits[-1]:.3f}, PSNR: {psnrs[-1]:.4f}, MS-SSIM: {msssims[-1]:.4f} ")

            if save_decoded_frame:
                save_path = os.path.join(args['decoded_frame_folder'], f'{frame_idx}.png')
                save_torch_image(x_hat, save_path)

    test_time = time.time() - start_time
    if verbose >= 1 and p_frame_number > 0:
        print(f"decoding {p_frame_number} P frames, "
              f"average {overall_p_decoding_time/p_frame_number * 1000:.0f} ms.")

    log_result = generate_log_json(frame_num, frame_types, bits, psnrs, msssims,
                                   frame_pixel_num, test_time)
    return log_result


def encode_one(args, device):
    i_state_dict = get_state_dict(args['i_frame_model_path'])
    i_frame_net = IntraNoAR() # 用于处理I帧的网络模型
    i_frame_net.load_state_dict(i_state_dict) # 加载预训练的模型参数
    i_frame_net = i_frame_net.to(device) # 将模型移动到指定的设备上
    i_frame_net.eval()

    if args['force_intra']:
        video_net = None
    else:
        p_state_dict = get_state_dict(args['model_path'])
        video_net = DMC()
        video_net.load_state_dict(p_state_dict)
        # # ========== 光流修正网络开始 ==========
        # # 添加 strict=False 兼容新旧模型参数
        # video_net.load_state_dict(p_state_dict, strict=False)
        # # 验证光流修正网络参数是否初始化
        # if hasattr(video_net.optic_flow, 'flow_correction'):
        #     print("光流修正网络已加载:", 
        #           video_net.optic_flow.flow_correction[0].weight.requires_grad)
        # # ========== 光流修正网络结束 ==========
        video_net = video_net.to(device)
        video_net.eval()

    if args['write_stream']:
        if video_net is not None:
            video_net.update(force=True)
        i_frame_net.update(force=True)

    sub_dir_name = args['video_path']
    gop_size = args['gop']
    frame_num = args['frame_num']

    bin_folder = os.path.join(args['stream_path'], sub_dir_name, str(args['rate_idx']))
    if args['write_stream']:
        create_folder(bin_folder, True)

    # 构建解码后的帧的保存路径
    if args['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args['decoded_frame_path'], sub_dir_name,
                                            str(args['rate_idx']))
        create_folder(decoded_frame_folder)
    else:
        decoded_frame_folder = None

    args['img_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    args['gop_size'] = gop_size
    args['frame_num'] = frame_num
    args['bin_folder'] = bin_folder
    args['decoded_frame_folder'] = decoded_frame_folder

    #* video_net = None：不使用帧间编码（P帧编码）；完整编码每一帧，压缩率较低；没有引入帧间编码的误差累积，视频质量高
    #* video_net != None：使用P帧编码；去除帧间冗余，提高压缩率；引入累计误差，视频质量较低
    #! 运行测试
    result = run_test(video_net, i_frame_net, args, device=device)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(use_cuda, args):
    # 环境设置
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)

    # GPU分配
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    #* 传入测试参数和设备信息进行编码
    result = encode_one(args, device)
    return result


def main():
    """
    主函数，负责执行整个测试流程。包括参数解析、路径检查、环境设置、多进程任务分配、结果收集和日志输出。
    
    流程概述：
    1. 记录开始时间。
    2. 启用CuDNN加速。
    3. 解析命令行参数。
    4. 检查模型路径是否存在。
    5. 设置CUDA设备和工作空间配置。
    6. 读取测试配置文件。
    7. 初始化多进程池并分配任务。
    8. 收集并处理测试结果。
    9. 输出测试结果到JSON文件。
    10. 计算并输出总测试时间和测试帧数。
    """
    begin_time = time.time()

    # 启用CuDNN加速
    torch.backends.cudnn.enabled = True
    args = parse_args()

    # 添加路径检查
    if not os.path.exists(args.i_frame_model_path):
        print(f"Error: i_frame_model_path {args.i_frame_model_path} does not exist.")
        return
    if not os.path.exists(args.model_path):
        print(f"Error: model_path {args.model_path} does not exist.")
        return

    # 设置CUDA设备和工作空间配置
    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    # 确保工作进程数（GPU数量）> 1
    worker_num = args.worker
    assert worker_num >= 1

    # 读取测试配置文件（dataset_config_example.json）
    with open(args.test_config) as f:
        config = json.load(f)

    # todo 参数化多进程池？
    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    count_frames = 0
    count_sequences = 0

    # 处理I帧的量化参数
    rate_num = args.rate_num
    i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)
    print("q_scales in intra ckpt: ", end='')
    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()
    if args.i_frame_q_scales is not None:
        assert len(args.i_frame_q_scales) == rate_num
        i_frame_q_scales = args.i_frame_q_scales
        print(f"testing {rate_num} rate points with pre-defined intra y q_scales: ", end='')
    elif len(i_frame_q_scales) == rate_num:
        print(f"testing {rate_num} rate points with intra y q_scales in ckpt: ", end='')
    else:
        max_q_scale = i_frame_q_scales[0]
        min_q_scale = i_frame_q_scales[-1]
        i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, rate_num)
        print(f"testing {rate_num} rates, using intra y q_scales: ", end='')

    for q in i_frame_q_scales:
        print(f"{q:.3f}, ", end='')
    print()

    # 处理P帧的量化参数
    if not args.force_intra:
        p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt(args.model_path)
        print("y_q_scales in inter ckpt: ", end='')
        for q in p_frame_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scales in inter ckpt: ", end='')
        for q in p_frame_mv_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        if args.p_frame_y_q_scales is not None:
            assert len(args.p_frame_y_q_scales) == rate_num
            assert len(args.p_frame_mv_y_q_scales) == rate_num
            p_frame_y_q_scales = args.p_frame_y_q_scales
            p_frame_mv_y_q_scales = args.p_frame_mv_y_q_scales
            print(f"testing {rate_num} rate points with pre-defined inter q_scales")
        elif len(p_frame_y_q_scales) == rate_num:
            print(f"testing {rate_num} rate points with inter q_scales in ckpt")
        else:
            max_y_q_scale = p_frame_y_q_scales[0]
            min_y_q_scale = p_frame_y_q_scales[-1]
            p_frame_y_q_scales = interpolate_log(min_y_q_scale, max_y_q_scale, rate_num)

            max_mv_y_q_scale = p_frame_mv_y_q_scales[0]
            min_mv_y_q_scale = p_frame_mv_y_q_scales[-1]
            p_frame_mv_y_q_scales = interpolate_log(min_mv_y_q_scale, max_mv_y_q_scale, rate_num)
        print("y_q_scales for testing: ", end='')
        for q in p_frame_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scales for testing: ", end='')
        for q in p_frame_mv_y_q_scales:
            print(f"{q:.3f}, ", end='')
        print()

    # 处理配置文件中的测试序列
    root_path = args.force_root_path if args.force_root_path is not None else config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            #* 分别对4个rate点进行测试，每个rate的具体数值由各个量化参数决定
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['i_frame_model_path'] = args.i_frame_model_path
                cur_args['i_frame_q_scale'] = i_frame_q_scales[rate_idx]
                if not args.force_intra:
                    cur_args['model_path'] = args.model_path
                    cur_args['p_frame_y_q_scale'] = p_frame_y_q_scales[rate_idx]
                    cur_args['p_frame_mv_y_q_scale'] = p_frame_mv_y_q_scales[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop'] = config[ds_name]['sequences'][seq_name]['gop']
                if args.force_intra:
                    cur_args['gop'] = 1
                if args.force_intra_period > 0:
                    cur_args['gop'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['decoded_frame_path'] = f'{args.decoded_frame_path}_DMC_{rate_idx}'
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['edge_alpha'] = args.edge_alpha # 新增边缘增强系数

                count_frames += cur_args['frame_num']

                #! 提交任务到进程池
                obj = threadpool_executor.submit( #* 异步方法，主线程会继续执行
                    worker, #* 会在一个新线程中执行worker函数
                    args.cuda,
                    cur_args)
                objs.append(obj)

    # 收集并处理测试结果
    results = []
    for obj in tqdm(objs):
        result = obj.result() #* 主线程会阻塞，等待任务执行结果
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][f"{rate:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
