import sys
from PyQt5.QtWidgets import QApplication
from hexchess_gui import MainWindow
import logging
import logging.config

def setup_logging(debug_mode=False):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': log_level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'level': log_level,
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': 'hexchess.log',
                'mode': 'w',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file'],
                'level': log_level,
                'propagate': True
            },
            'hexchess': {
                'handlers': ['default', 'file'],
                'level': log_level,
                'propagate': False
            },
            'hexchess.ai': {
                'handlers': ['default', 'file'],
                'level': log_level,
                'propagate': False
            },
        }
    })



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    setup_logging(debug_mode=True)  # Set to False for production
    main()
