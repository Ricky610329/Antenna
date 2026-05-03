# ? python -m script.process_files -t main.py (依賴模式)
# ? python -m script.process_files -dir ./ (目錄掃描模式)
import argparse
import ast
import importlib.util
import os
import sys
from collections import deque
from datetime import datetime
from io import TextIOWrapper

from loguru import logger
from tqdm import tqdm


class FileProcessor:
    # 類別層級的常數，作為預設忽略模式
    IGNORED_PATTERNS = {
        "__pycache__",
        ".git",
        "venv",
        ".vscode",
        ".idea",
        ".env",
        "node_modules",
        "build",
        "dist",
        "script",
        "result",
        "docs",
        "abandon",
        "wandb",
    }

    def __init__(
        self,
        directories: str | list[str] | tuple[str] = ".",
        extensions: list[str] = [".py"],
        output_file: str = "project_report.txt",
        output_dir: str = ".",
        project_name: str = "專案報告",
        project_description: str | None = None,
        generated_by: str | None = None,
        verbose: bool = True,
    ):
        """
        :param directories: 要掃描或限制分析範圍的資料夾路徑。
        :param extensions: 要包含的副檔名。
        :param generated_by: (依賴模式用) 指定入口檔案路徑。若為 None，則執行全目錄掃描。
        """
        self.directories = directories if isinstance(directories, (list, tuple)) else [directories]
        # 預先計算絕對路徑，用於嚴格過濾檔案範圍
        self.abs_directories = [os.path.abspath(d) for d in self.directories]

        self.extensions = extensions
        self.output_file = output_file
        self.output_dir = output_dir
        self.extensions_tuple = tuple(self.extensions)
        self.project_name = project_name

        self.generated_by = os.path.abspath(generated_by) if generated_by else None

        self.project_description = project_description
        self.verbose = verbose
        self.scanned_files: list[str] = []

    # ================= 依賴分析相關方法 (Dependency Mode) =================

    def _is_custom_module(self, file_path: str) -> bool:
        """判斷檔案是否為自定義模組（非系統路徑且位於指定目錄內）。"""
        try:
            if not file_path:
                return False
            abs_path = os.path.abspath(file_path)

            # 1. 系統 / 虛擬環境路徑排除
            if any(x in abs_path for x in ["site-packages", "dist-packages", "venv", ".env", "__pycache__"]):
                return False
            if "lib/python" in abs_path.replace("\\", "/"):
                return False

            # 2. 透過 commonpath 檢查檔案是否在允許的目錄範圍內
            for d in self.abs_directories:
                try:
                    if os.path.commonpath([d, abs_path]) == d:
                        return True
                except ValueError:
                    # Windows 跨磁碟機時 commonpath 會拋 ValueError
                    continue
            return False
        except Exception:
            return False

    def _resolve_import_path(self, module_name: str, base_dir: str, level: int = 0) -> str | None:
        """嘗試解析 import 的實際檔案路徑。"""
        try:
            # 相對路徑 (from . import x)
            if level > 0:
                package_parts = module_name.split(".") if module_name else []
                current_dir = base_dir
                for _ in range(level - 1):
                    current_dir = os.path.dirname(current_dir)

                if not package_parts:
                    return None

                guess_path = os.path.join(current_dir, *package_parts)
                candidates = [guess_path + ".py", os.path.join(guess_path, "__init__.py")]
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
                rel_path = module_name.replace(".", os.sep)
                candidates = [
                    os.path.join(base_dir, rel_path + ".py"),
                    os.path.join(base_dir, rel_path, "__init__.py"),
                    os.path.join(os.getcwd(), rel_path + ".py"),
                ]
                for c in candidates:
                    if os.path.exists(c) and os.path.isfile(c):
                        return os.path.abspath(c)

        except Exception:
            pass
        return None

    def _analyze_file_imports(self, file_path: str) -> set[str]:
        """解析單一檔案的 AST 並回傳所有發現的自定義依賴檔案路徑。"""
        found_files = set()
        if not os.path.exists(file_path):
            return found_files

        try:
            with open(file_path, encoding="utf-8") as f:
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
                        if alias.name == "*":
                            continue
                        sub_module_name = f"{module_name}.{alias.name}" if module_name else alias.name
                        path = self._resolve_import_path(sub_module_name, base_dir, level)
                        if path and self._is_custom_module(path):
                            found_files.add(path)
        return found_files

    def _scan_dependencies_recursive(self, entry_file: str) -> list[str]:
        """遞迴掃描依賴 (BFS)。"""
        visited = {entry_file}
        queue = deque([entry_file])
        result_files = []
        pbar = tqdm(desc="Analyzing Dependencies", disable=not self.verbose)

        while queue:
            current_file = queue.popleft()

            # 只有當檔案位於指定目錄內時才納入報告；
            # 但即使入口檔不在範圍內，仍會分析其 imports 以找出範圍內的依賴。
            if self._is_custom_module(current_file):
                result_files.append(current_file)
            elif current_file == entry_file and self.verbose:
                logger.info(
                    f"注意: 入口檔案 {os.path.basename(current_file)} 不在指定目錄內，將不會包含其內容，僅分析其依賴。"
                )

            pbar.update(1)
            pbar.set_postfix(file=os.path.basename(current_file))

            for imp in self._analyze_file_imports(current_file):
                if imp not in visited:
                    visited.add(imp)
                    queue.append(imp)

        pbar.close()
        return result_files

    # ================= 目錄掃描相關方法 (Directory Mode) =================

    def _scan_directories_recursive(self) -> list[str]:
        """掃描 directories 中符合 extension 的所有檔案。"""
        files_found = []
        if self.verbose:
            logger.info(f"開始掃描目錄: {self.directories}")

        for d in self.directories:
            if not os.path.exists(d):
                continue
            walk_iter = os.walk(d)
            for root, dirs, files in walk_iter:
                dirs[:] = [d for d in dirs if d not in self.IGNORED_PATTERNS]
                for file in files:
                    if file.endswith(self.extensions_tuple):
                        files_found.append(os.path.join(root, file))
        return files_found

    def _generate_tree_section(self, outfile: TextIOWrapper):
        outfile.write(f"\n\n{'=' * 20} 專案目錄結構 (Directory Tree) {'=' * 20}\n")

        # 用 Set 快速查詢哪些檔案已納入報告
        scanned_set = {os.path.abspath(f) for f in self.scanned_files}

        for directory_path in self.directories:
            abs_dir_path = os.path.abspath(directory_path)
            outfile.write("\n附註: (*) 代表該檔案已被納入報告內容。\n")
            outfile.write(f"Root: {os.path.basename(abs_dir_path)} ({abs_dir_path})\n")
            self._create_tree_recursive(outfile, abs_dir_path, "", scanned_set)

    def _create_tree_recursive(
        self, file_obj: TextIOWrapper, dir_path: str, prefix: str = "", scanned_set: set[str] = None
    ):
        if scanned_set is None:
            scanned_set = set()

        try:
            items = [item for item in os.listdir(dir_path) if item not in self.IGNORED_PATTERNS]
            entries = sorted(items, key=lambda x: not os.path.isdir(os.path.join(dir_path, x)))
        except OSError:
            return

        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            full_path = os.path.join(dir_path, entry)
            # 檔案若已納入報告則加上 (*) 標記
            is_included = os.path.abspath(full_path) in scanned_set
            marker = " (*)" if is_included and os.path.isfile(full_path) else ""

            file_obj.write(f"{prefix}{connector}{entry}{marker}\n")

            if os.path.isdir(full_path):
                new_prefix = "    " if i == len(entries) - 1 else "│   "
                self._create_tree_recursive(file_obj, full_path, prefix + new_prefix, scanned_set)

    # ================= 共用報告生成方法 =================

    def _write_main_header(self, file_obj: TextIOWrapper):
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
        outfile.write(f"\n\n{'=' * 20} 包含的檔案清單 (File List) {'=' * 20}\n")
        try:
            common_path = os.path.commonpath(self.scanned_files)
        except (ValueError, OSError):
            # 跨磁碟機或路徑無共通祖先時 fallback 到 cwd
            common_path = os.getcwd()
        outfile.write(f"Common Root: {common_path}\n")
        for f in sorted(self.scanned_files):
            outfile.write(f" - {os.path.relpath(f, common_path)}\n")

    def _combine_files_section(self, outfile: TextIOWrapper):
        sorted_files = sorted(list(self.scanned_files))
        _tqdm = tqdm(sorted_files, desc="Writing content", disable=not self.verbose)

        outfile.write(f"\n\n{'=' * 20} 檔案內容 (File Contents) {'=' * 20}\n")
        for file_path in _tqdm:
            outfile.write(f"\n{'=' * 30} FILE: {file_path} {'=' * 30}\n")
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")
            except Exception as e:
                outfile.write(f"*** 讀取錯誤: {e} ***\n")

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"無法建立輸出資料夾: {e}")
            return

        if self.generated_by:
            if self.verbose:
                logger.info(f"[Mode] 依賴分析模式 (Recursive): {self.generated_by}")
            if not os.path.exists(self.generated_by):
                logger.error(f"找不到入口檔案: {self.generated_by}")
                return
            self.scanned_files = self._scan_dependencies_recursive(self.generated_by)
        else:
            if self.verbose:
                logger.info(f"[Mode] 目錄掃描模式: {self.directories}")
            self.scanned_files = self._scan_directories_recursive()

        output_path = os.path.join(self.output_dir, self.output_file)
        if self.verbose:
            logger.info(f"找到 {len(self.scanned_files)} 個符合條件的檔案，正在寫入報告...")

        try:
            with open(output_path, "w", encoding="utf-8") as outfile:
                self._write_main_header(outfile)
                self._generate_tree_section(outfile)
                if self.generated_by:
                    self._generate_file_list_section(outfile)
                self._combine_files_section(outfile)
            logger.success(f"完成！報告位置: {output_path}")

        except Exception as e:
            logger.error(f"錯誤: {e}")
            import traceback

            logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(
        description="專案代碼整合報告生成器", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-t", "--target", default=None, help="[依賴模式] 指定入口檔案。")
    parser.add_argument("-d", "--directories", default=["."], nargs="+", help="分析範圍資料夾 (預設: .)。")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".py"], help="副檔名 (預設: .py)。")
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
        generated_by=args.target,
    )
    processor.run()


if __name__ == "__main__":
    # 若無提供 CLI 參數則以預設（全目錄掃描）執行
    if len(sys.argv) > 1:
        main()
    else:
        FileProcessor(generated_by=None, verbose=True).run()
