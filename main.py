import sys
import os
import logging

from config_types import load_pipeline_config
from processor import DatasetProcessor

def main():
    """主入口函数"""
    logger = logging.getLogger("tts_preprocess")
    try:
        cfg = load_pipeline_config()
        processor = DatasetProcessor(cfg)
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
