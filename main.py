from core.mock_ga_runner import MockGAService
from gui.app import Application

def main():
    ga_service = MockGAService()
    Application(ga_service)

if __name__ == '__main__':
    main()