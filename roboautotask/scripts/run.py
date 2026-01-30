import sys
import argparse
import logging_mp

from roboautotask.core.operator import Operator


logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


def main():
    # 解析命令行参数

    parser = argparse.ArgumentParser(description='自动化采集任务脚本')
    
    parser.add_argument('--task-id', type=int, required=True,
                       help='任务ID（必填）')
    parser.add_argument('--user', type=str, required=True,
                       help='登录账号（必填）')
    parser.add_argument('--password', type=str, required=True,
                       help='登录密码（必填）')
    parser.add_argument('--url', type=str, default='http://localhost:5805/hmi/login',
                       help='登录URL，默认为 http://localhost:5805/hmi/login')
    parser.add_argument('--headless', action='store_true',
                       help='以无头模式运行（不显示浏览器窗口）')
    parser.add_argument('--task-wait-timeout', type=int, default=10000,
                       help='等待元素加载的超时时间（毫秒），默认为10000')
    args = parser.parse_args()
    
    logger.info(f"启动参数:")
    logger.info(f"  任务ID: {args.task_id}")
    logger.info(f"  用户名: {args.user}")
    logger.info(f"  登录URL: {args.url}")
    logger.info(f"  无头模式: {'是' if args.headless else '否'}")
    
    operator = Operator(args)

    operator.login()

if __name__ == "__main__":
    main()