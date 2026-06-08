# -*- coding: utf-8 -*-
#? python -m script.process_files -t main.py (依賴模式)
#? python -m script.process_files -dir ./ (目錄掃描模式)
# ==============================================================================
# FileProcessor — 實驗可重現性核心工具
#
# 【用途】
#   天線逆設計流程中，每次實驗啟動時 get_result_path() 會呼叫本模組，
#   將「產生這次實驗的訓練腳本及其所有自訂依賴」一次性快照至結果資料夾。
#   這樣即使日後修改程式碼，仍可從結果資料夾還原當時的確切程式狀態。
#
# 【兩種操作模式】
#   1. 依賴分析模式 (Dependency Mode)：
#      指定一個入口腳本（如 train_single.py），以 BFS 遞迴解析其
#      import 依賴，只收集「位於指定目錄範圍內」的自訂模組，
#      排除標準庫與第三方套件，輸出一份彙整報告。
#
#   2. 全目錄掃描模式 (Directory Scan Mode)：
#      直接走訪指定目錄樹，蒐集所有符合副檔名的檔案，
#      不做 import 追蹤，適合快速打包整個子目錄。
#
# 【輸出內容】
#   單一 UTF-8 文字檔，依序包含：
#     - 標頭（專案名稱、時間戳、模式、說明）
#     - 目錄樹（標記哪些檔案已納入報告）
#     - 依賴模式才有：檔案清單（相對路徑）
#     - 所有納入檔案的完整原始碼
# ==============================================================================
import os
import argparse
import sys
import ast
import importlib.util
from datetime import datetime
from typing import Union, Optional, List, Tuple, Set
from io import TextIOWrapper
from loguru import logger
from tqdm import tqdm

class FileProcessor:
    """
    程式碼快照產生器，供 get_result_path() 在建立實驗目錄時呼叫。

    核心價值：每次實驗都在結果資料夾留下「當下程式碼快照」，
    讓日後可以不依賴 git log 直接重現訓練環境。

    兩種模式由 generated_by 參數切換：
      - generated_by 非 None → 依賴分析模式（追蹤 import 圖）
      - generated_by 為 None → 全目錄掃描模式（走訪資料夾）
    """
    # 類別層級的常數，作為預設忽略模式
    #! 這些目錄永遠被跳過（含目錄掃描與依賴過濾兩個階段）
    #* 加入此集合即可全域排除不需要快照的路徑（如 result/、wandb/ 等大型輸出目錄）
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
        'wandb',
    }

    def __init__(
            self,
            directories: Union[str, List[str], Tuple[str]] = '.',
            extensions: List[str] = ['.py'],
            output_file: str = "project_report.txt",
            output_dir: str = ".",
            project_name: str = "專案報告",
            project_description: Optional[str] = None,
            generated_by: Optional[str] = None,
            verbose: bool = True
        ):
        """
        :param directories: 要掃描或限制分析範圍的資料夾路徑。
        :param extensions: 要包含的副檔名。
        :param generated_by: (依賴模式用) 指定入口檔案路徑。若為 None，則執行全目錄掃描。
        """
        # directories 接受字串或清單，統一正規化成 list
        self.directories = directories if isinstance(directories, (list, tuple)) else [directories]
        # [新增] 預先計算絕對路徑，用於嚴格過濾檔案範圍
        #* 後續 _is_custom_module() 以此判斷模組是否在「允許範圍」內
        self.abs_directories = [os.path.abspath(d) for d in self.directories]

        self.extensions = extensions
        self.output_file = output_file            # 輸出報告的檔名
        self.output_dir = output_dir              # 輸出報告的目錄（即實驗結果資料夾）
        self.extensions_tuple = tuple(self.extensions)
        self.project_name = project_name          # 寫入報告標頭的顯示名稱

        # generated_by：觸發本次快照的訓練腳本絕對路徑；None 代表目錄掃描模式
        self.generated_by = os.path.abspath(generated_by) if generated_by else None

        self.project_description = project_description
        self.verbose = verbose
        self.scanned_files: List[str] = []        # run() 執行後，存放所有納入報告的檔案路徑

    # ================= 依賴分析相關方法 (Dependency Mode) =================

    def _is_custom_module(self, file_path: str) -> bool:
        """
        判斷檔案是否為自定義模組。
        條件：
        1. 非 site-packages/venv 等系統目錄。
        2. [新增] 必須位於指定的 directories 範圍內。
        """
        # 此方法是依賴過濾的守門員：
        #   - 排除標準庫/第三方套件（site-packages、lib/python 等）
        #   - 排除不在 directories 範圍內的路徑
        # 通過兩道關卡才視為「需要納入快照的自訂模組」
        try:
            if not file_path: return False
            abs_path = os.path.abspath(file_path)

            # 1. 系統/環境路徑排除
            if any(x in abs_path for x in ['site-packages', 'dist-packages', 'venv', '.env', '__pycache__']):
                return False
            if "lib/python" in abs_path.replace("\\", "/"):
                return False

            # 2. [嚴格過濾] 檢查檔案是否在允許的目錄清單中
            # 使用 commonpath 判斷層級關係
            in_scope = False
            for d in self.abs_directories:
                try:
                    # 如果 d 是 abs_path 的父目錄(或相同)，commonpath 結果應為 d
                    if os.path.commonpath([d, abs_path]) == d:
                        in_scope = True
                        break
                except ValueError:
                    # 處理 Windows 不同磁碟機的情況
                    continue

            if not in_scope:
                # logger.debug(f"跳過範圍外檔案: {abs_path}")
                return False

            return True
        except Exception:
            return False

    def _resolve_import_path(self, module_name: str, base_dir: str, level: int = 0) -> Optional[str]:
        """嘗試解析 import 的實際檔案路徑。"""
        # 解析策略（優先順序）：
        #   1. 相對 import（level > 0，如 from . import x）→ 依 level 回溯目錄層級後手動組路徑
        #   2. 絕對 import → 利用 importlib.util.find_spec 取得 spec.origin
        #   3. Fallback → 對 base_dir 與 cwd 手動拼接 .py / __init__.py 候選路徑
        # 為了讓 importlib 能找到當前目錄下的模組，暫時將 base_dir 加入 path
        search_path = [base_dir, os.getcwd()] + sys.path

        try:
            # 相對路徑 (from . import x)
            if level > 0:
                package_parts = module_name.split('.') if module_name else []
                current_dir = base_dir
                for _ in range(level - 1):
                    current_dir = os.path.dirname(current_dir)

                if not package_parts: return None

                guess_path = os.path.join(current_dir, *package_parts)
                candidates = [guess_path + '.py', os.path.join(guess_path, '__init__.py')]
                for c in candidates:
                    if os.path.exists(c) and os.path.isfile(c):
                        return os.path.abspath(c)

            # 絕對路徑 / 標準解析
            if level == 0:
                try:
                    original_sys_path = sys.path[:]
                    sys.path.insert(0, base_dir)
                    spec = importlib.util.find_spec(module_name)
                    sys.path = original_sys_path
                    if spec and spec.origin:
                        return spec.origin
                except (ImportError, ValueError, AttributeError):
                    pass

            # Fallback
            if level == 0:
                rel_path = module_name.replace('.', os.sep)
                candidates = [
                    os.path.join(base_dir, rel_path + '.py'),
                    os.path.join(base_dir, rel_path, '__init__.py'),
                    os.path.join(os.getcwd(), rel_path + '.py'),
                ]
                for c in candidates:
                    if os.path.exists(c) and os.path.isfile(c):
                        return os.path.abspath(c)

        except Exception:
            pass
        return None

    def _analyze_file_imports(self, file_path: str) -> Set[str]:
        """解析單一檔案的 AST 並回傳所有發現的自定義依賴檔案路徑。"""
        # 做法：以 ast.walk 遍歷語法樹，對每個 Import / ImportFrom 節點
        #   呼叫 _resolve_import_path() 取得實體路徑，
        #   再以 _is_custom_module() 過濾，只保留在 directories 範圍內的自訂模組。
        # ImportFrom 額外嘗試解析 `from pkg import submod` 的子模組路徑，
        #   以免 `from antenna import loss` 中的 loss.py 被漏掉。
        found_files = set()
        if not os.path.exists(file_path): return found_files

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=file_path)
        except Exception as e:
            logger.warning(f"無法解析 AST: {file_path} ({e})")
            return found_files

        base_dir = os.path.dirname(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    path = self._resolve_import_path(alias.name, base_dir, level=0)
                    if path and self._is_custom_module(path):
                        found_files.add(path)

            elif isinstance(node, ast.ImportFrom):
                level = node.level if node.level is not None else 0
                module_name = node.module

                if module_name or level > 0:
                    path = self._resolve_import_path(module_name, base_dir, level)
                    if path and self._is_custom_module(path):
                        found_files.add(path)

                if node.names:
                    for alias in node.names:
                        if alias.name == '*': continue
                        sub_module_name = f"{module_name}.{alias.name}" if module_name else alias.name
                        path = self._resolve_import_path(sub_module_name, base_dir, level)
                        if path and self._is_custom_module(path):
                            found_files.add(path)
        return found_files

    def _scan_dependencies_recursive(self, entry_file: str) -> List[str]:
        """遞迴掃描依賴 (BFS)。"""
        # BFS 從 entry_file 出發，逐層展開 import 圖：
        #   - 每個節點呼叫 _analyze_file_imports() 取得直接依賴
        #   - 透過 visited set 避免循環引用
        #   - 只有通過 _is_custom_module() 的節點才進入 result_files
        #   - 入口腳本本身若在範圍外（如絕對路徑的訓練腳本），
        #     仍會被分析其依賴，但其原始碼不納入報告
        visited = set()
        queue = [entry_file]
        visited.add(entry_file)

        result_files = []
        pbar = tqdm(desc="Analyzing Dependencies", disable=not self.verbose)

        while queue:
            current_file = queue.pop(0)

            # [修改] 只有當檔案位於指定目錄內時，才加入最終報告清單
            # 注意：即使檔案不在目錄內（例如外部入口腳本），我們仍然會分析其依賴，
            # 以便找出它是否引用了目錄內的檔案。
            if self._is_custom_module(current_file):
                result_files.append(current_file)
            elif current_file == entry_file and self.verbose:
                logger.info(f"注意: 入口檔案 {os.path.basename(current_file)} 不在指定目錄內，將不會包含其內容，僅分析其依賴。")

            pbar.update(1)
            pbar.set_postfix(file=os.path.basename(current_file))

            # 找出此檔案的 imports
            new_imports = self._analyze_file_imports(current_file)

            for imp in new_imports:
                if imp not in visited:
                    visited.add(imp)
                    queue.append(imp)

        pbar.close()
        return result_files

    # ================= 目錄掃描相關方法 (Directory Mode) =================

    def _scan_directories_recursive(self) -> List[str]:
        """掃描 directories 中符合 extension 的所有檔案。"""
        # 以 os.walk 走訪目錄樹，動態剪枝（dirs[:] = ...）跳過 IGNORED_PATTERNS，
        # 避免進入 result/、wandb/ 等無需快照的輸出目錄。
        files_found = []
        if self.verbose:
            logger.info(f"開始掃描目錄: {self.directories}")

        for d in self.directories:
            if not os.path.exists(d): continue
            walk_iter = os.walk(d)
            for root, dirs, files in walk_iter:
                dirs[:] = [d for d in dirs if d not in self.IGNORED_PATTERNS]
                for file in files:
                    if file.endswith(self.extensions_tuple):
                        files_found.append(os.path.join(root, file))
        return files_found

    def _generate_tree_section(self, outfile: TextIOWrapper):
        """輸出目錄樹，並以 (*) 標記已納入報告的檔案，便於一眼確認快照涵蓋範圍。"""
        outfile.write(f"\n\n{'=' * 20} 專案目錄結構 (Directory Tree) {'=' * 20}\n")

        # [修改] 建立一個 Set 來快速查詢哪些檔案已被納入分析
        scanned_set = {os.path.abspath(f) for f in self.scanned_files}

        for directory_path in self.directories:
            abs_dir_path = os.path.abspath(directory_path)
            outfile.write(f"\n附註: (*) 代表該檔案已被納入報告內容。\n")
            outfile.write(f"Root: {os.path.basename(abs_dir_path)} ({abs_dir_path})\n")
            self._create_tree_recursive(outfile, abs_dir_path, "", scanned_set)

    def _create_tree_recursive(self, file_obj: TextIOWrapper, dir_path: str, prefix: str = "", scanned_set: Set[str] = None):
        """遞迴繪製目錄樹（類 Unix tree 格式）；scanned_set 用於標記已納入報告的檔案。"""
        if scanned_set is None: scanned_set = set()

        try:
            items = [item for item in os.listdir(dir_path) if item not in self.IGNORED_PATTERNS]
            entries = sorted(items, key=lambda x: not os.path.isdir(os.path.join(dir_path, x)))
        except OSError:
            return

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "

            # [修改] 檢查檔案是否在 scanned_set 中，若是則加上標記
            full_path = os.path.join(dir_path, entry)
            is_included = os.path.abspath(full_path) in scanned_set
            marker = " (*)" if is_included and os.path.isfile(full_path) else ""

            file_obj.write(f"{prefix}{connector}{entry}{marker}\n")

            entry_path = os.path.join(dir_path, entry)
            if os.path.isdir(entry_path):
                new_prefix = "    " if i == len(entries) - 1 else "│   "
                self._create_tree_recursive(file_obj, entry_path, prefix + new_prefix, scanned_set)

    # ================= 共用報告生成方法 =================

    def _write_main_header(self, file_obj: TextIOWrapper):
        """寫入報告標頭：專案名、時間戳、模式、說明、納入檔案數量。"""
        # 時間戳讓讀者知道快照是何時產生的，
        # 配合 get_result_path() 在實驗開始時呼叫，
        # 時間戳會與實驗結果資料夾名稱大致對應。
        file_obj.write(f"Project Name: {self.project_name}\n")
        file_obj.write(f"Generate Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if self.generated_by:
            mode_str = f"依賴分析模式 (Dependency Mode) - Entry: {os.path.basename(self.generated_by)}"
            desc = self.project_description or "此報告基於入口檔案進行依賴分析，僅包含位於指定目錄內的相關程式碼。"
        else:
            mode_str = "全目錄掃描模式 (Directory Scan Mode)"
            desc = self.project_description or "此報告包含指定目錄下的所有符合副檔名之檔案。"

        file_obj.write(f"Mode: {mode_str}\n")
        file_obj.write(f"Target Directories: {', '.join(self.directories)}\n")
        file_obj.write(f"Description: {desc}\n")
        file_obj.write(f"Total Files Included: {len(self.scanned_files)}\n")
        file_obj.write("=" * 80 + "\n")

    def _generate_file_list_section(self, outfile: TextIOWrapper):
        """依賴模式專用：輸出納入報告的所有檔案的相對路徑清單，便於快速瀏覽依賴圖涵蓋範圍。"""
        outfile.write(f"\n\n{'=' * 20} 包含的檔案清單 (File List) {'=' * 20}\n")
        try:
            common_path = os.path.commonpath(self.scanned_files)
        except:
            common_path = os.getcwd()
        outfile.write(f"Common Root: {common_path}\n")
        for f in sorted(self.scanned_files):
            outfile.write(f" - {os.path.relpath(f, common_path)}\n")

    def _combine_files_section(self, outfile: TextIOWrapper):
        """將所有納入報告的檔案原始碼依路徑排序後，逐一附加到報告末尾。"""
        # 每個檔案前後以分隔線標示路徑，讓報告可直接按 FILE: 關鍵字搜尋。
        # errors='ignore' 容錯非 UTF-8 字元（如舊版 Windows 路徑名稱混入 cp950 編碼時）。
        sorted_files = sorted(list(self.scanned_files))
        _tqdm = tqdm(sorted_files, desc="Writing content", disable=not self.verbose)

        outfile.write(f"\n\n{'=' * 20} 檔案內容 (File Contents) {'=' * 20}\n")
        for file_path in _tqdm:
            outfile.write(f"\n{'='*30} FILE: {file_path} {'='*30}\n")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")
            except Exception as e:
                outfile.write(f"*** 讀取錯誤: {e} ***\n")

    def run(self):
        """
        主流程入口，依 generated_by 決定模式後執行快照，輸出報告。

        流程：
          1. 建立 output_dir（即實驗結果資料夾，已存在也不報錯）
          2. 收集檔案清單（依賴模式 BFS / 目錄掃描模式 walk）
          3. 依序寫入：標頭 → 目錄樹 → 檔案清單（依賴模式）→ 原始碼內容
          4. 完成後以 logger.success 回報報告路徑

        可重現性意義：
          run() 在實驗開始時被呼叫，報告與模型 checkpoint 儲存在同一個結果目錄，
          任何時候都能從這份報告還原當時的程式碼狀態，無需依賴外部版本控制。
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"無法建立輸出資料夾: {e}")
            return

        if self.generated_by:
            if self.verbose: logger.info(f"[Mode] 依賴分析模式 (Recursive): {self.generated_by}")
            if not os.path.exists(self.generated_by):
                logger.error(f"找不到入口檔案: {self.generated_by}")
                return
            self.scanned_files = self._scan_dependencies_recursive(self.generated_by)
        else:
            if self.verbose: logger.info(f"[Mode] 目錄掃描模式: {self.directories}")
            self.scanned_files = self._scan_directories_recursive()

        output_path = os.path.join(self.output_dir, self.output_file)
        if self.verbose: logger.info(f"找到 {len(self.scanned_files)} 個符合條件的檔案，正在寫入報告...")

        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                self._write_main_header(outfile)
                self._generate_tree_section(outfile)
                if self.generated_by: self._generate_file_list_section(outfile)
                self._combine_files_section(outfile)
            logger.success(f"完成！報告位置: {output_path}")

        except Exception as e:
            logger.error(f"錯誤: {e}")
            import traceback
            logger.error(traceback.format_exc())

def main():
    """CLI 入口點，供直接以 python -m script.process_files 呼叫。"""
    parser = argparse.ArgumentParser(description="專案代碼整合報告生成器", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-t", "--target", default=None, help="[依賴模式] 指定入口檔案。")
    parser.add_argument("-d", "--directories", default=['.'], nargs='+', help="分析範圍資料夾 (預設: .)。")
    parser.add_argument("-e", "--extensions", nargs='+', default=['.py'], help="副檔名 (預設: .py)。")
    parser.add_argument("-o", "--output", default="project_report.txt", help="輸出檔名。")
    parser.add_argument("-od", "--output_dir", default=".", help="輸出資料夾。")
    parser.add_argument("-n", "--name", default="Project Report", help="專案名稱。")
    args = parser.parse_args()

    processor = FileProcessor(
        directories=args.directories,
        extensions=args.extensions,
        output_file=args.output,
        output_dir=args.output_dir,
        project_name=args.name,
        generated_by=args.target
    )
    processor.run()

if __name__ == "__main__":
    FileProcessor(
        generated_by=None,#r'C:\timmy\Program\Antenna\train_dual.py',
        verbose = True
    ).run()
