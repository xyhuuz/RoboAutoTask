import sys
import cv2
import argparse
import logging_mp
import time
import numpy as np

from roboautotask.core.operator import Operator
from roboautotask.core.motion import MotionExecutor

from roboautotask.utils.pose import save_pose_to_file

from roboautotask.configs.robot import ROBOT_START_POS, ROBOT_START_ORI


logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


def main():
    # 解析命令行参数

    parser = argparse.ArgumentParser(description='自动化采集任务脚本')
    
    parser.add_argument('--motions', type=str, required=True,
                       help='运动配置yaml文件路径')
    parser.add_argument('--task-id', type=int, required=True,
                       help='任务ID（必填）')
    parser.add_argument('--user', type=str, required=True,
                       help='登录账号（必填）')
    parser.add_argument('--password', type=str, required=True,
                       help='登录密码（必填）')
    parser.add_argument('--url', type=str, default='http://localhost:5805/hmi/login',
                       help='登录URL，默认为 http://localhost:5805/hmi/login')
    parser.add_argument('--headless', action='store_true', default=False,
                       help='以无头模式运行（不显示浏览器窗口）')
    parser.add_argument('--task-wait-timeout', type=int, default=10000,
                       help='等待元素加载的超时时间（毫秒），默认为10000')
    args = parser.parse_args()
    
    logger.info(f"启动参数:")
    logger.info(f"  任务ID: {args.task_id}")
    logger.info(f"  用户名: {args.user}")
    logger.info(f"  登录URL: {args.url}")
    logger.info(f"  无头模式: {'是' if args.headless else '否'}")


    save_pose_to_file("./latest_pose.txt", ROBOT_START_POS, ROBOT_START_ORI)

    operator = Operator(args)
    motion_executor = MotionExecutor(args.motions)

    operator.login()

    # try:
    #     while True:
    #         operator.find_task()
    #         operator.start_task()

    #         ### 执行采集任务
    #         motion_sequence = [2, -3, 0]

    #         for sid in motion_sequence:
    #             if not motion_executor.execute_by_id(sid):
    #                 logger.info(f"Sequence aborted at ID {sid}")
    #                 break

    #         operator.complete_task()
    #         operator.commit_task()
    #         operator.quit_task()

    #         logger.info("任务采集完成")

    #         ### 执行场景重置
    #         # motion_sequence = [2, 1, 0]

    #         for sid in motion_sequence:
    #             if not motion_executor.reset(sid):
    #                 logger.info(f"Sequence aborted at reset")
    #                 break
            
    #         logger.info("场景重置完成")

    #         time.sleep(3)
    try:
        while True:
            ### 执行采集任务
            motion_sequence = [[2, -3]]

            # 这是抓取&放置一套流程；抓取物&放置物并发识别；
            for sid in motion_sequence:
                grab_id,place_id = sid

                operator.find_task()
                operator.exec_task()
                operator.start_task()

                result = motion_executor.execute_by_id(grab_id, place_id)
                if result == 0:
                    logger.info(f"Sequence aborted at ID {sid}")
                    break
                elif result == 2:
                    logger.info('无法采集，需要重置！')
                    operator.destroy_task()
                    operator.quit_task()

                    if not motion_executor.reset(grab_id,place_id):
                        logger.info(f"Sequence aborted at reset")
                    logger.info("场景重置完成")
                    break

                elif result == 3:
                    logger.info('运动时间超时，需要检查目标物位置！')
                    operator.destroy_task()
                    operator.quit_task()
                    # 加按键等待
                    # --- 新增：CV2 按键等待功能 ---
                    # 创建一个黑色的画布
                    img = np.zeros((400, 800, 3), np.uint8)
                    # 添加提示文字
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, 'TIMEOUT: Check object position!', (50, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, 'Press [C] to Continue / [ESC] to Quit', (50, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.namedWindow('Robot_Alert', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Robot_Alert', img)
                    
                    logger.info("等待用户按键反馈...")
                    
                    while True:
                        # 等待按键输入，1ms检查一次
                        key = cv2.waitKey(1) & 0xFF
                        
                        # 如果按下 'c' 或 'C'
                        if key == ord('c') or key == ord('C'):
                            logger.info("用户按下 C，准备重新开始循环...")
                            cv2.destroyWindow('Robot_Alert')
                            break # 跳出当前的 key 等待循环
                        
                        # 如果按下 ESC
                        elif key == 27:
                            logger.info("用户按下 ESC，终止任务")
                            cv2.destroyWindow('Robot_Alert')
                            raise SystemExit # 直接退出程序

                    # continue 将会跳过本次 while True 的剩余逻辑，进入下一次任务循环
                    continue



                operator.complete_task()
                operator.commit_task()
                operator.quit_task()

                logger.info("任务采集完成")

                ### 执行场景重置
                # motion_sequence = [2, 1, 0]

                if not motion_executor.reset(grab_id,place_id):
                    logger.info(f"Sequence aborted at reset")
                    break
                
                logger.info("场景重置完成")

                time.sleep(3)

    finally:
        operator.stop()
        motion_executor.go_home()


if __name__ == "__main__":
    main()
