# -*- coding: utf-8 -*-
#? python -m script.process_files ./ -e .py
import os
import argparse
from datetime import datetime

# --- 全域設定: 在此處新增您想在「文件樹」中忽略的資料夾或檔案名稱 ---
# 使用集合(set)可以提高查找效率
IGNORED_PATTERNS = {
    '__pycache__',  # Python 的快取資料夾 (使用者要求)
    '.git',         # Git 版本控制資料夾
    'venv',         # Python 虛擬環境資料夾
    '.vscode',      # VSCode 編輯器設定資料夾
    '.idea',        # JetBrains IDEs 設定資料夾
    '.env',         # 環境變數檔案
    'node_modules', # Node.js 依賴庫資料夾
    'build',        # 編譯輸出資料夾
    'dist',         # 打包輸出資料夾
    'script', 'result', 'docs', 'abandon' # Antenna
}


def combine_files(directory_paths, extensions, output_file_path):
    """
    遞迴掃描多個指定目錄，將所有符合指定副檔名列表的檔案整合成單一文件。
    """
    extensions_tuple = tuple(extensions)
    print(f"[*] 開始整合副檔名為 '{', '.join(extensions)}' 的檔案...")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            # 寫入總報告頭
            outfile.write(f"檔案整合報告\n")
            outfile.write(f"目標目錄: {', '.join([os.path.abspath(p) for p in directory_paths])}\n")
            outfile.write(f"目標副檔名: {', '.join(extensions)}\n")
            outfile.write(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            outfile.write("="*50 + "\n")

            # 遍歷每一個指定的目錄
            for directory_path in directory_paths:
                abs_dir_path = os.path.abspath(directory_path)
                print(f"\n[+] 正在處理目錄: {abs_dir_path}")
                outfile.write(f"\n\n{'='*20} 開始處理目錄: {abs_dir_path} {'='*20}\n\n")

                found_files = False
                for root, _, files in os.walk(directory_path):
                    # 忽略被指定的資料夾
                    if any(ignored in root.split(os.sep) for ignored in IGNORED_PATTERNS):
                        continue
                        
                    for file in sorted(files):
                        if file.endswith(extensions_tuple):
                            found_files = True
                            file_path = os.path.join(root, file)
                            print(f"  -> 正在讀取: {file_path}")
                            outfile.write(f"----------- {file_path} -----------\n\n")
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                                    content = infile.read()
                                    outfile.write(content)
                                    outfile.write("\n\n")
                            except Exception as e:
                                outfile.write(f"*** 無法讀取檔案: {e} ***\n\n")
                
                if not found_files:
                    print(f"  -> 在 {abs_dir_path} 中未找到符合副檔名的檔案。")
                    outfile.write(f"*** 在此目錄中未找到符合副檔名 '{', '.join(extensions)}' 的檔案 ***\n\n")


        print(f"\n[+] 成功！所有檔案已整合至: {output_file_path}\n")

    except Exception as e:
        print(f"[!] 整合檔案時發生錯誤: {e}\n")


def generate_tree(directory_paths, output_file_path, ignored=None):
    """
    為多個指定目錄產生文件樹結構，並將所有樹狀圖儲存至單一 txt 檔。
    """
    if ignored is None:
        ignored = set()
    
    print(f"[*] 開始產生文件樹 (將忽略: {', '.join(ignored)})...")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"多目錄文件樹報告\n")
            f.write(f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if ignored:
                f.write(f"(全域忽略規則: {', '.join(sorted(list(ignored)))})\n")
            f.write("="*50 + "\n")

            # 遍歷每一個指定的目錄
            for directory_path in directory_paths:
                abs_dir_path = os.path.abspath(directory_path)
                print(f"\n[+] 正在為目錄產生樹狀圖: {abs_dir_path}")
                f.write(f"\n\n{'='*20} 目錄樹: {abs_dir_path} {'='*20}\n")
                
                f.write(f"{os.path.basename(abs_dir_path)}\n")
                _create_tree_recursive(f, abs_dir_path, "", ignored)
        
        print(f"\n[+] 成功！所有文件樹已儲存至: {output_file_path}\n")

    except Exception as e:
        print(f"[!] 產生文件樹時發生錯誤: {e}\n")


def _create_tree_recursive(file_obj, dir_path, prefix="", ignored=None):
    """
    遞迴輔助函式，用來建立文件樹的每一層結構，會跳過忽略列表中的項目。
    (此函式功能不變)
    """
    if ignored is None:
        ignored = set()

    try:
        # 取得目錄下所有項目，並預先過濾掉要忽略的檔案/資料夾
        items = [item for item in os.listdir(dir_path) if item not in ignored]
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
            _create_tree_recursive(file_obj, entry_path, prefix + new_prefix, ignored)


def main():
    """
    主函式，處理使用者輸入並執行主要功能。
    """
    parser = argparse.ArgumentParser(
        description="一個強大的文件處理工具，可以整合指定副檔名的檔案，並為多個目錄產生文件樹。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "directories", 
        nargs='+',  # 接受一個或多個值
        help="要處理的一個或多個目標資料夾路徑，請用空格分隔。"
    )
    parser.add_argument(
        "-e", "--extensions", 
        nargs='+',
        default=['.php', '.css'], 
        help="要整合的一個或多個檔案副檔名，請用空格分隔。\n範例: -e .py .html .css (預設值: .py)"
    )
    parser.add_argument("-o1", "--output_combine", default="combine_files.txt", help="整合後的文件名稱。(預設值: combine_files.txt)")
    parser.add_argument("-o2", "--output_tree", default="trees.txt", help="文件樹的輸出文件名稱。(預設值: trees.txt)")
    
    args = parser.parse_args()

    directory_paths = args.directories
    extensions = args.extensions
    output_combine_path = args.output_combine
    output_tree_path = args.output_tree

    # 驗證所有提供的路徑都是有效的資料夾
    for path in directory_paths:
        if not os.path.isdir(path):
            print(f"[!] 錯誤：指定的路徑 '{path}' 不是一個有效的資料夾或不存在。")
            return

    # 傳入目錄列表進行處理
    combine_files(directory_paths, extensions, output_combine_path)
    generate_tree(directory_paths, output_tree_path, ignored=IGNORED_PATTERNS)
    
    print("✨ 所有任務執行完畢！ ✨")

#python process_files.py app\Admin app\Announcement app\Auth app\Config app\Controllers app\Database app\Filters app\Models app\Views public

if __name__ == "__main__":
    main()