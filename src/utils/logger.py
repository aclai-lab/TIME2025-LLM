from utils.ansi import AnsiCodes


class Logger:
    @staticmethod
    def info(source: str, message: str, verbose: bool = True, faint: bool = True):
        if not verbose: return
        faint_tag = AnsiCodes.FAINT if faint else ''
        print(f'{AnsiCodes.BOLD}{faint_tag}[INFO]{AnsiCodes.RESET}{faint_tag} {source} | {message}{AnsiCodes.RESET}')

    @staticmethod
    def warning(source: str, message: str, verbose: bool = True, faint: bool = True):
        if not verbose: return
        faint_tag = AnsiCodes.FAINT if faint else ''
        print(f'{AnsiCodes.BOLD}{faint_tag}{AnsiCodes.FG_YELLOW}[WARNING]{AnsiCodes.RESET}{AnsiCodes.FG_YELLOW}{faint_tag} {source} | {message}{AnsiCodes.RESET}')

    @staticmethod
    def error(source: str, message: str, faint: bool = False):
        faint_tag = AnsiCodes.FAINT if faint else ''
        print(f'{AnsiCodes.BOLD}{faint_tag}{AnsiCodes.FG_RED}[ERROR]{AnsiCodes.RESET}{AnsiCodes.FG_RED}{faint_tag} {source} | {message}{AnsiCodes.RESET}')
