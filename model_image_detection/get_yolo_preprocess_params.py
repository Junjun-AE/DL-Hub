"""
获取 Ultralytics YOLO 实际预处理参数的脚本

运行方法:
    pip install ultralytics
    python get_yolo_preprocess_params.py

此脚本会输出 YOLO 实际使用的预处理参数，用于更新 checkpoint.py
"""

import numpy as np
import torch


def method1_check_predictor_source():
    """方法1: 查看 BasePredictor.preprocess 源码"""
    print("=" * 70)
    print("方法1: 查看 BasePredictor.preprocess() 源码")
    print("=" * 70)
    
    try:
        from ultralytics.engine.predictor import BasePredictor
        import inspect
        
        source = inspect.getsource(BasePredictor.preprocess)
        print(source)
        print()
    except Exception as e:
        print(f"错误: {e}")
        print()


def method2_check_transforms():
    """方法2: 检查模型的 transforms 属性"""
    print("=" * 70)
    print("方法2: 检查模型的 transforms 属性")
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        
        model = YOLO('yolov8n.pt')
        
        print(f"model.transforms: {model.transforms}")
        
        if hasattr(model.model, 'transforms'):
            print(f"model.model.transforms: {model.model.transforms}")
        
        # 检查模型内部
        if hasattr(model, 'predictor'):
            print(f"model.predictor: {model.predictor}")
        
        print()
    except Exception as e:
        print(f"错误: {e}")
        print()


def method3_trace_preprocess():
    """方法3: 实际跟踪预处理过程"""
    print("=" * 70)
    print("方法3: 实际跟踪预处理过程")
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        import cv2
        
        model = YOLO('yolov8n.pt')
        
        # 创建测试图像: 全255 (白色)
        test_img_white = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        # 创建测试图像: 全0 (黑色)
        test_img_black = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # 创建测试图像: 特定值 [100, 150, 200]
        test_img_rgb = np.zeros((640, 640, 3), dtype=np.uint8)
        test_img_rgb[:, :, 0] = 100  # R
        test_img_rgb[:, :, 1] = 150  # G
        test_img_rgb[:, :, 2] = 200  # B
        
        # 保存临时图像
        cv2.imwrite('/tmp/test_white.jpg', test_img_white)
        cv2.imwrite('/tmp/test_black.jpg', test_img_black)
        cv2.imwrite('/tmp/test_rgb.jpg', cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2BGR))
        
        # 钩子函数来捕获预处理后的张量
        preprocessed_tensors = []
        
        def hook_preprocess(predictor):
            original_preprocess = predictor.preprocess
            
            def wrapped_preprocess(im):
                result = original_preprocess(im)
                preprocessed_tensors.append(result.clone())
                return result
            
            predictor.preprocess = wrapped_preprocess
        
        # 运行预测并检查
        print("\n测试1: 白色图像 (255, 255, 255)")
        results = model.predict('/tmp/test_white.jpg', verbose=False)
        # 获取预处理后的张量
        if hasattr(results[0], 'orig_img'):
            print(f"  原始图像形状: {results[0].orig_img.shape}")
        
        print("\n测试2: 黑色图像 (0, 0, 0)")
        results = model.predict('/tmp/test_black.jpg', verbose=False)
        
        print("\n测试3: RGB图像 (100, 150, 200)")
        results = model.predict('/tmp/test_rgb.jpg', verbose=False)
        
        print()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print()


def method4_manual_preprocess_test():
    """方法4: 手动复现预处理并验证"""
    print("=" * 70)
    print("方法4: 手动复现预处理并验证")
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox
        import cv2
        
        model = YOLO('yolov8n.pt')
        
        # 创建测试图像
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:, :, 0] = 100  # B (OpenCV是BGR)
        test_img[:, :, 1] = 150  # G
        test_img[:, :, 2] = 200  # R
        
        cv2.imwrite('/tmp/test_manual.jpg', test_img)
        
        print("\n原始像素值 (BGR): B=100, G=150, R=200")
        
        # 使用 LetterBox
        letterbox = LetterBox(new_shape=(640, 640))
        img_letterbox = letterbox(image=test_img)
        print(f"Letterbox后形状: {img_letterbox.shape}")
        
        # 检查填充区域的值
        print(f"Letterbox填充颜色 (如果有填充): {img_letterbox[0, 0, :]}")
        
        # 模拟YOLO预处理
        img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = img_float.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        
        print(f"\n预处理后:")
        print(f"  形状: {img_tensor.shape}")
        print(f"  数据类型: {img_tensor.dtype}")
        print(f"  最小值: {img_tensor.min().item():.4f}")
        print(f"  最大值: {img_tensor.max().item():.4f}")
        print(f"  中心像素值 (R, G, B): {img_tensor[0, :, 320, 320].tolist()}")
        
        # 验证计算
        print(f"\n验证:")
        print(f"  原始R=200 → 200/255 = {200/255:.4f}")
        print(f"  原始G=150 → 150/255 = {150/255:.4f}")
        print(f"  原始B=100 → 100/255 = {100/255:.4f}")
        
        print()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print()


def method5_check_augment_source():
    """方法5: 查看数据增强源码"""
    print("=" * 70)
    print("方法5: 查看数据加载器预处理源码")
    print("=" * 70)
    
    try:
        # 查看 BaseTransform
        from ultralytics.data.augment import BaseTransform, LetterBox, Format
        import inspect
        
        print("\n--- LetterBox ---")
        print(inspect.getsource(LetterBox.__init__))
        
        print("\n--- Format (转换为tensor) ---")
        if hasattr(Format, '__call__'):
            print(inspect.getsource(Format.__call__))
        
    except Exception as e:
        print(f"错误: {e}")
        print()


def method6_check_dataset_loader():
    """方法6: 检查数据加载器的预处理"""
    print("=" * 70)
    print("方法6: 检查 YOLODataset 预处理")
    print("=" * 70)
    
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.build import build_dataloader
        import inspect
        
        # 查看数据集的 transforms
        print("\n查看 YOLODataset.build_transforms 方法:")
        if hasattr(YOLODataset, 'build_transforms'):
            print(inspect.getsource(YOLODataset.build_transforms))
        
    except Exception as e:
        print(f"错误: {e}")
        print()


def method7_definitive_test():
    """方法7: 最终确认测试"""
    print("=" * 70)
    print("方法7: 最终确认 - 比较YOLO内部预处理与手动预处理")
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        import cv2
        
        model = YOLO('yolov8n.pt')
        
        # 创建特殊测试图像
        # 用特定像素值来验证是否有额外的 mean/std 归一化
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_img[300:340, 300:340, :] = [128, 128, 128]  # 中心灰色方块
        
        cv2.imwrite('/tmp/test_final.jpg', test_img)
        
        # 方法A: 使用YOLO内部处理
        # 获取推理时的实际输入张量
        
        class TensorCapture:
            def __init__(self):
                self.captured = None
        
        capture = TensorCapture()
        
        # Hook into the model
        original_forward = model.model.forward
        
        def hooked_forward(x, *args, **kwargs):
            capture.captured = x.clone()
            return original_forward(x, *args, **kwargs)
        
        model.model.forward = hooked_forward
        
        # 运行预测
        results = model.predict('/tmp/test_final.jpg', verbose=False)
        
        # 恢复原始forward
        model.model.forward = original_forward
        
        if capture.captured is not None:
            tensor = capture.captured
            print(f"\nYOLO内部预处理后的张量:")
            print(f"  形状: {tensor.shape}")
            print(f"  数据类型: {tensor.dtype}")
            print(f"  最小值: {tensor.min().item():.6f}")
            print(f"  最大值: {tensor.max().item():.6f}")
            print(f"  均值: {tensor.mean().item():.6f}")
            
            # 检查中心灰色方块区域 (应该是128/255 ≈ 0.502)
            center_values = tensor[0, :, 310:330, 310:330].mean(dim=(1, 2))
            print(f"  中心灰色方块 (应该≈0.502):")
            print(f"    R通道均值: {center_values[0].item():.6f}")
            print(f"    G通道均值: {center_values[1].item():.6f}")
            print(f"    B通道均值: {center_values[2].item():.6f}")
            
            # 检查黑色背景区域 (应该是0)
            bg_values = tensor[0, :, 0:100, 0:100].mean(dim=(1, 2))
            print(f"  黑色背景 (应该≈0):")
            print(f"    R通道均值: {bg_values[0].item():.6f}")
            print(f"    G通道均值: {bg_values[1].item():.6f}")
            print(f"    B通道均值: {bg_values[2].item():.6f}")
            
            # 判断归一化方式
            print(f"\n结论:")
            if abs(center_values[0].item() - 0.502) < 0.01:
                print("  ✓ 确认: YOLO 只做 /255 归一化，不做 ImageNet mean/std")
                print("  ✓ normalize_mean = (0.0, 0.0, 0.0)")
                print("  ✓ normalize_std = (1.0, 1.0, 1.0)")
                print("  ✓ value_range = (0.0, 1.0)")
            else:
                # 如果使用了ImageNet归一化
                # (128/255 - 0.485) / 0.229 ≈ 0.074
                expected_imagenet = (128/255 - 0.485) / 0.229
                if abs(center_values[0].item() - expected_imagenet) < 0.1:
                    print("  ! 发现: YOLO 使用了 ImageNet mean/std 归一化")
                    print("  ! normalize_mean = (0.485, 0.456, 0.406)")
                    print("  ! normalize_std = (0.229, 0.224, 0.225)")
                else:
                    print(f"  ? 未知归一化方式，请检查值: {center_values.tolist()}")
        else:
            print("未能捕获张量")
        
        print()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print()


def main():
    """运行所有测试方法"""
    print("\n" + "=" * 70)
    print("Ultralytics YOLO 预处理参数获取工具")
    print("=" * 70)
    
    import ultralytics
    print(f"\nUltralytics 版本: {ultralytics.__version__}")
    print()
    
    # 运行各种方法
    method1_check_predictor_source()
    method2_check_transforms()
    method4_manual_preprocess_test()
    method7_definitive_test()
    
    print("\n" + "=" * 70)
    print("测试完成！请根据上面的输出确认预处理参数")
    print("=" * 70)


if __name__ == '__main__':
    main()
