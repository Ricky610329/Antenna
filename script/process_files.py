# -*- coding: utf-8 -*-
#? python -m script.process_files ./ -e .py
import os
import argparse
from datetime import datetime
from loguru import logger
from typing import Union, Optional
from tqdm import tqdm
###* Type ###
from _io import TextIOWrapper
class FileProcessor:
    """
    一個強大的文件處理工具，可以整合指定副檔名的檔案，
    並為多個目錄產生文件樹。
    """
    
    # 類別層級的常數，作為預設忽略模式
    IGNORED_PATTERNS = {
        '__pycache__', 
        '.git',
        'venv',
        '.vscode',
        '.idea',
        '.env',
        'node_modules',
        'build',
        'dist',
        'script', 'result', 'docs', 'abandon'
    }

    def __init__(
            self, 
            directories:Union[str, list, tuple] = '.', 
            extensions:list = ['.py'], 
            output_combine:str = "combine_files.txt", 
            output_tree:str = "trees.txt", 
            output_dir:str = ".", 
            project_name:str = "純生成無專案", 
            project_description:Optional[str] = None, 
            generated_by:str = __file__
        ):
        """
        初始化 FileProcessor 實例。

        :param directories: 要處理的一個或多個目標資料夾路徑。
        :param extensions: 要整合的一個或多個檔案副檔名。
        :param output_combine: 整合後的文件名稱。
        :param output_tree: 文件樹的輸出文件名稱。
        :param output_dir: 所有輸出文件的目標資料夾。
        :param project_name: 報告中顯示的專案名稱。
        :param project_description: 報告中顯示的專案描述。
        :param generated_by: 產生此報告的主程式名稱。

        Example::

            processor = FileProcessor(
                directories=ROOTDIR,
                extensions=['.py'],
                output_dir = ROOTDIR,
                project_description = "",
                generated_by=__file__
            )
            processor.run()
        """
        self.directories = directories if isinstance(directories, (list, tuple)) else [directories]
        self.extensions = extensions
        self.output_combine = output_combine
        self.output_tree = output_tree
        self.output_dir = output_dir
        self.extensions_tuple = tuple(self.extensions)
        self.project_name = project_name
        self.generated_by = generated_by
        self.project_description = project_description or "此報告包含程式碼檔案的整合內容與專案文件樹結構，用於提供專案的完整概覽。"
      
    def _validate_directories(self):
        """
        驗證所有指定的目錄路徑是否有效，並創建輸出資料夾。
        """
        for path in self.directories:
            if not os.path.isdir(path):
                logger.error(f"指定的路徑 '{path}' 不是一個有效的資料夾或不存在。")
                return False
        
        # 確保輸出資料夾存在
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"輸出資料夾 '{self.output_dir}' 已確認或建立。")
        except OSError as e:
            logger.error(f"無法建立輸出資料夾 '{self.output_dir}': {e}")
            return False

        return True

    def _write_header(self, file_obj:TextIOWrapper, title:str, specific_info:str=""):
        """
        寫入統一的報告頭部資訊。
        """
        file_obj.write(f"Project Name: {self.project_name}\n")
        file_obj.write(f"Title: {title}\n")
        file_obj.write(f"Project Description{self.project_description}\n")
        file_obj.write("=" * 50 + "\n")
        file_obj.write(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_obj.write(f"目標目錄: {', '.join([os.path.abspath(p) for p in self.directories])}\n")
        file_obj.write(f"忽略規則: {', '.join(sorted(list(self.IGNORED_PATTERNS)))}\n")
        if self.generated_by:
            file_obj.write(f"報告生成者:  此報告由 `{self.generated_by}` 為主程式生成。\n")
        if specific_info:
            file_obj.write(f"{specific_info}\n")
        file_obj.write("=" * 50 + "\n\n")

    def combine_files(self):
        """
        遞迴掃描多個指定目錄，將所有符合指定副檔名列表的檔案整合成單一文件。
        """
        logger.info(f"開始整合副檔名為 '{', '.join(self.extensions)}' 的檔案...")
        
        # 組成完整的輸出路徑
        output_path = os.path.join(self.output_dir, self.output_combine)

        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                # 寫入總報告頭
                specific_info = f"**目標副檔名**: {', '.join(self.extensions)}"
                self._write_header(outfile, "檔案整合報告", specific_info)

                for directory_path in self.directories:
                    abs_dir_path = os.path.abspath(directory_path)
                    logger.success(f"正在處理目錄: {abs_dir_path}")
                    outfile.write(f"\n\n{'=' * 20} 開始處理目錄: {abs_dir_path} {'=' * 20}\n\n")

                    found_files = False
                    _tqdm = tqdm(os.walk(directory_path), desc="Reading...")
                    for root, dirs, files in _tqdm:
                        # 忽略指定的資料夾，並在遍歷前修改 dirs 列表
                        dirs[:] = [d for d in dirs if d not in self.IGNORED_PATTERNS]

                        for file in sorted(files):
                            if file.endswith(self.extensions_tuple):
                                found_files = True
                                file_path = os.path.join(root, file)
                                # logger.info(f"  -> 正在讀取: {file_path}")
                                _tqdm.set_postfix({"File": file_path})
                                outfile.write(f"----------- {file_path} -----------\n\n")
                                try:
                                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                        content = infile.read()
                                        outfile.write(content)
                                        outfile.write("\n\n")
                                except Exception as e:
                                    outfile.write(f"*** 無法讀取檔案: {e} ***\n\n")
                    
                    if not found_files:
                        logger.info(f"  -> 在 {abs_dir_path} 中未找到符合副檔名的檔案。")
                        outfile.write(f"*** 在此目錄中未找到符合副檔名 '{', '.join(self.extensions)}' 的檔案 ***\n\n")

            logger.success(f"成功！所有檔案已整合至: {output_path}\n")

        except Exception as e:
            logger.error(f"整合檔案時發生錯誤: {e}\n")
    
    def generate_tree(self):
        """
        為多個指定目錄產生文件樹結構，並將所有樹狀圖儲存至單一 txt 檔。
        """
        logger.info(f"開始產生文件樹 (將忽略: {', '.join(self.IGNORED_PATTERNS)})...")
        
        # 組成完整的輸出路徑
        output_path = os.path.join(self.output_dir, self.output_tree)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # 寫入總報告頭
                self._write_header(f, "文件樹報告")

                for directory_path in self.directories:
                    abs_dir_path = os.path.abspath(directory_path)
                    logger.success(f"正在為目錄產生樹狀圖: {abs_dir_path}")
                    f.write(f"\n\n{'=' * 20} 目錄樹: {abs_dir_path} {'=' * 20}\n")
                    
                    f.write(f"{os.path.basename(abs_dir_path)}\n")
                    self._create_tree_recursive(f, abs_dir_path, "")
            
            logger.success(f"成功！所有文件樹已儲存至: {output_path}\n")

        except Exception as e:
            logger.error(f"產生文件樹時發生錯誤: {e}\n")

    def _create_tree_recursive(self, file_obj:TextIOWrapper, dir_path:str, prefix:str=""):
        """
        遞迴輔助函式，用來建立文件樹的每一層結構，會跳過忽略列表中的項目。
        """
        try:
            # 取得目錄下所有項目，並預先過濾掉要忽略的檔案/資料夾
            items = [item for item in os.listdir(dir_path) if item not in self.IGNORED_PATTERNS]
            # 將資料夾排在前面
            entries = sorted(items, key=lambda x: not os.path.isdir(os.path.join(dir_path, x)))
        except OSError as e:
            file_obj.write(f"{prefix}└── [無法存取: {e}]\n")
            return

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            file_obj.write(f"{prefix}{connector}{entry}\n")

            entry_path = os.path.join(dir_path, entry)
            if os.path.isdir(entry_path):
                new_prefix = "    " if i == len(entries) - 1 else "│   "
                self._create_tree_recursive(file_obj, entry_path, prefix + new_prefix)
    
    def run(self):
        """
        執行所有主要任務。
        """
        if not self._validate_directories():
            return
        
        self.combine_files()
        self.generate_tree()
        logger.success("所有任務執行完畢！ ✨")

def main():
    """
    主函式，處理使用者輸入並建立 FileProcessor 實例。
    """
    parser = argparse.ArgumentParser(
        description="一個強大的文件處理工具，可以整合指定副檔名的檔案，並為多個目錄產生文件樹。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-dir", "--directories", 
        default=['.'],
        nargs='+', 
        help="要處理的一個或多個目標資料夾路徑，請用空格分隔。"
    )
    parser.add_argument(
        "-e", "--extensions", 
        nargs='+',
        default=['.py'], 
        help="要整合的一個或多個檔案副檔名，請用空格分隔。\n範例: -e .py .html .css (預設值: .py)"
    )
    parser.add_argument("-o1", "--output_combine", default="combine_files.txt", help="整合後的文件名稱。(預設值: combine_files.txt)")
    parser.add_argument("-o2", "--output_tree", default="trees.txt", help="文件樹的輸出文件名稱。(預設值: trees.txt)")
    parser.add_argument("-o3", "--output_dir", default=".", help="所有輸出文件的目標資料夾。(預設值: .)")
    parser.add_argument("-n", "--project_name", default="Project Report", help="設定報告中的專案名稱。")
    parser.add_argument("-d", "--project_description", default="", help="設定報告中的專案描述。")
    
    args = parser.parse_args()

    import sys
    # 建立 FileProcessor 實例
    processor = FileProcessor(
        directories=args.directories,
        extensions=args.extensions,
        output_combine=args.output_combine,
        output_tree=args.output_tree,
        output_dir=args.output_dir,
        project_name=args.project_name,
        project_description=args.project_description,
        generated_by= f"Command({' '.join(sys.argv)})"
    )

    # 執行任務
    processor.run()


if __name__ == "__main__":
    main()