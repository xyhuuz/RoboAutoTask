import re
import time

from playwright.sync_api import Playwright, sync_playwright, expect, Browser
import logging_mp
from dataclasses import dataclass

logger = logging_mp.get_logger(__name__)

@dataclass
class Operator:
    """平台网页自动操作器"""
    def __init__(self, args):
        self.playwright: Playwright = sync_playwright().start()
        self.headless = args.headless
        self.browser: Browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context()

        self.url = args.url
        self.user = args.user
        self.password = args.password

        self.task = None
        self.task_id = args.task_id
        self.task_wait_timeout = args.task_wait_timeout
        
    def login(self):
        self.page = self.context.new_page()

        # 登录平台
        logger.info(f"正在访问: {self.url}")
        self.page.goto(self.url)
        
        logger.info(f"正在登录，用户名: {self.user}")
        self.page.get_by_role("textbox", name="请输入账号").click()
        self.page.get_by_role("textbox", name="请输入账号").fill(self.user)
        
        self.page.get_by_role("textbox", name="请输入登录密码").click()
        self.page.get_by_role("textbox", name="请输入登录密码").fill(self.password)
        
        self.page.get_by_role("button", name="登录").click()

        time.sleep(2)

    def find_task(self):
        # 查找指定任务ID的行并点击采集按钮
        try:
            logger.info(f"正在查找任务ID: {self.task_id}")
            # 等待表格加载
            self.page.wait_for_selector(".el-table__row", timeout=self.task_wait_timeout)
            
            # 找到所有行
            rows = self.page.locator(".el-table__row").all()
            target_row = None
            
            for row in rows:
                # 获取该行的第一列（ID列）的文本
                id_cell = row.locator(".el-table_1_column_1 .cell").first
                if id_cell.count() > 0:
                    cell_text = id_cell.inner_text().strip()
                    if cell_text == str(self.task_id):
                        target_row = row
                        break
            
            if target_row:
                logger.info(f"✓ 找到任务ID: {self.task_id}")
                # 在该行中查找"采集"按钮
                capture_button = target_row.locator("button.btn-capture")
                if capture_button.count() > 0:
                    capture_button.click()
                    logger.info(f"✓ 已点击任务 {self.task_id} 的采集按钮")
                    time.sleep(2)
                else:
                    logger.info(f"✗ 任务 {self.task_id} 未找到采集按钮（可能状态不可采集）")
            else:
                logger.info(f"✗ 未找到任务ID: {self.task_id}")
                
        except Exception as e:
            logger.error(f"✗ 查找任务时出错: {e}")
        
        logger.info("等待10秒后关闭...")
        time.sleep(10)


    def start_task(self):
        pass

    def complete_task():
        pass
    
    def stop(self) -> None:
        self.context.close()
        self.browser.close()
