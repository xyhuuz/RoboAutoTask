from roboautotask.scripts import reset



def main():
    reset.logger.info('RoboAutoTask正在启动')
    reset.main()
    reset.logger.info('RoboAutoTask启动完成')


if __name__ == "__main__":
    main()
