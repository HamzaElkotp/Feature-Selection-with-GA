from core.GA_API_run import GA_Service
from gui.app import Application

def main():
    ga_service = GA_Service()
    Application(ga_service)

if __name__ == '__main__':
    main()