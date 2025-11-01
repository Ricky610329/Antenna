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
import os
from datetime import datetime
from typing import Union, Optional, List, Tuple
from io import TextIOWrapper
from tqdm import tqdm
from loguru import logger # 假設您使用 loguru，如果不是請替換

class FileProcessor:
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
        'script', 'result', 'docs', 'abandon',
        'wandb'
    }

    def __init__(
            self, 
            directories: Union[str, List[str], Tuple[str]] = '.', 
            extensions: List[str] = ['.py'], 
            output_file: str = "project_report.txt",
            output_dir: str = ".", 
            project_name: str = "純生成無專案", 
            project_description: Optional[str] = None, 
            generated_by: str = __file__,
            verbose: bool = True 
        ):
        """
        將多個目錄的原始碼與文件樹整合到單一報告檔案中。

        :param directories: 要處理的一個或多個目標資料夾路徑。
        :param extensions: 要整合的一個或多個檔案副檔名。
        :param output_file: 整合後的單一報告文件名稱。
        :param output_dir: 所有輸出文件的目標資料夾。
        :param project_name: 報告中顯示的專案名稱。
        :param project_description: 報告中顯示的專案描述。
        :param generated_by: 產生此報告的主程式名稱。
        :param verbose: 是否在控制台顯示日誌和進度條。
        """
        self.directories = directories if isinstance(directories, (list, tuple)) else [directories]
        self.extensions = extensions
        self.output_file = output_file
        self.output_dir = output_dir
        self.extensions_tuple = tuple(self.extensions)
        self.project_name = project_name
        self.generated_by = generated_by
        self.project_description = project_description or "此報告包含程式碼檔案的整合內容與專案文件樹結構，用於提供專案的完整概覽。"
        self.verbose = verbose

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
            if self.verbose:
                logger.info(f"輸出資料夾 '{self.output_dir}' 已確認或建立。")
        except OSError as e:
            if self.verbose:
                logger.error(f"無法建立輸出資料夾 '{self.output_dir}': {e}")
            return False

        return True

    def _write_main_header(self, file_obj: TextIOWrapper):
        """
        在單一報告文件的頂部寫入總標頭。
        """
        file_obj.write(f"Project Name: {self.project_name}\n")
        file_obj.write(f"Title: 專案總報告 (Project Report)\n")
        file_obj.write(f"Project Description: {self.project_description}\n")
        file_obj.write("=" * 80 + "\n")
        file_obj.write(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_obj.write(f"目標目錄: {', '.join([os.path.abspath(p) for p in self.directories])}\n")
        file_obj.write(f"目標副檔名: {', '.join(self.extensions)}\n")
        file_obj.write(f"忽略規則: {', '.join(sorted(list(self.IGNORED_PATTERNS)))}\n")
        if self.generated_by:
            file_obj.write(f"報告生成者: 此報告由專案主程式 `{self.generated_by}` 產生。\n")
        file_obj.write("=" * 80 + "\n")

    def _combine_files_section(self, outfile: TextIOWrapper):
        """
        將檔案內容寫入到提供的文件物件中（作為報告的一個區段）。
        """
        if self.verbose:  # <--- 修改點 3: 加入 verbose 判斷
            logger.info(f"開始整合副檔名為 '{', '.join(self.extensions)}' 的檔案...")
        
        for directory_path in self.directories:
            abs_dir_path = os.path.abspath(directory_path)
            if self.verbose:  # <--- 修改點 3: 加入 verbose 判斷
                logger.success(f"正在處理目錄: {abs_dir_path}")

            found_files = False
            
            # <--- 修改點 4: 控制 tqdm 的 disable 狀態 ---
            walk_iter = os.walk(directory_path)
            _tqdm = tqdm(
                walk_iter, 
                desc="Reading...", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}", 
                disable=not self.verbose
            )
            
            for root, dirs, files in _tqdm:
                # 忽略指定的資料夾
                dirs[:] = [d for d in dirs if d not in self.IGNORED_PATTERNS]

                for file in sorted(files):
                    if file.endswith(self.extensions_tuple):
                        found_files = True
                        file_path = os.path.join(root, file)
                        
                        if self.verbose:  # <--- 修改點 5: tqdm 更新也需控制
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
                if self.verbose:
                    logger.info(f"  -> 在 {abs_dir_path} 中未找到符合副檔名的檔案。")
                outfile.write(f"*** 在此目錄中未找到符合副檔名 '{', '.join(self.extensions)}' 的檔案 ***\n\n")

        if self.verbose:  # <--- 修改點 3: 加入 verbose 判斷
            logger.info("...檔案整合部分已完成。")

    def _generate_tree_section(self, outfile: TextIOWrapper):
        """
        將文件樹結構寫入到提供的文件物件中（作為報告的一個區段）。
        """
        if self.verbose:
            logger.info(f"開始產生文件樹 (將忽略: {', '.join(self.IGNORED_PATTERNS)})...")
        
        for directory_path in self.directories:
            abs_dir_path = os.path.abspath(directory_path)
            if self.verbose:
                logger.success(f"正在為目錄產生樹狀圖: {abs_dir_path}")
            outfile.write(f"\n\n{'=' * 20} 目錄樹: {abs_dir_path} {'=' * 20}\n")
            
            outfile.write(f"{os.path.basename(abs_dir_path)}\n")
            self._create_tree_recursive(outfile, abs_dir_path, "")
        
        if self.verbose:
            logger.info("...文件樹部分已完成。")

    def _create_tree_recursive(self, file_obj: TextIOWrapper, dir_path: str, prefix: str = ""):
        """
        遞迴輔助函式，用來建立文件樹的每一層結構，會跳過忽略列表中的項目。
        """
        try:
            items = [item for item in os.listdir(dir_path) if item not in self.IGNORED_PATTERNS]
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
        執行所有主要任務，並將所有結果寫入單一報告文件。
        """
        if not self._validate_directories():
            return
        
        output_path = os.path.join(self.output_dir, self.output_file)
        if self.verbose:  # <--- 修改點 3: 加入 verbose 判斷
            logger.info(f"開始產生總報告: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                # 1. 寫入總標頭
                self._write_main_header(outfile)
                
                # 2. 寫入文件樹部分
                outfile.write("\n\n\n" + "I. 文件樹 (Directory Tree)" + "\n" + "="*80 + "\n")
                self._generate_tree_section(outfile)

                # 3. 寫入檔案整合部分
                outfile.write("\n\n" + "II. 檔案整合 (File Contents)" + "\n" + "="*80 + "\n")
                self._combine_files_section(outfile)
                
            logger.success(f"專案報告已成功寫入: {output_path}")

        except Exception as e:
            if self.verbose:
                logger.error(f"產生總報告時發生錯誤: {e}")
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
        # directories=args.directories,
        # extensions=args.extensions,
        # output_combine=args.output_combine,
        # output_tree=args.output_tree,
        # output_dir=args.output_dir,
        # project_name=args.project_name,
        # project_description=args.project_description,
        generated_by= f"Command({' '.join(sys.argv)})"
    )

    # 執行任務
    processor.run()


if __name__ == "__main__":
    main()