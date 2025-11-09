import sys
import yaml
import os
import logging

from processor import DatasetProcessor

def load_config(config_path='config.yaml'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主入口函数"""
    logger = logging.getLogger("tts_preprocess")
    try:
        # 从 config.yaml 加载配置
        config = load_config()
        
        # 直接将整个配置字典传递给处理器
        processor = DatasetProcessor(config)
        processor.run()
        
    except MemoryError as e:
        error_msg = "发生内存错误 (MemoryError)。这很可能是因为加载模型所需的内存（RAM或VRAM）超出了系统可用资源。请尝试关闭其他占用大量内存的程序，或在资源更丰富的环境中运行。"
        print(f"\n[FATAL] {error_msg}", file=sys.stderr)
        if logger.handlers:
            logger.error(error_msg, exc_info=True)
        sys.exit(1)
    except Exception as e:
        error_msg = f"发生未处理的异常: {e}"
        print(f"\n[FATAL] {error_msg}", file=sys.stderr)
        # 如果日志记录器已配置，则记录完整的异常追溯信息
        if logger.handlers:
            logger.error("发生未处理的异常", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
