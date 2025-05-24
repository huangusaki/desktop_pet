import os
import tokenize
import io
import sys


def remove_comments_from_file(filepath):
    """
    从指定的 Python 文件中移除所有 # 注释。
    Args:
        filepath (str): 要处理的文件的完整路径。
    """
    try:
        # 使用 'utf-8' 编码打开文件
        with open(filepath, "r", encoding="utf-8") as f:
            source_code = f.read()
        # 将源代码字符串包装成一个文件对象，供 tokenize 使用
        source_code_io = io.StringIO(source_code)
        # 生成 token 列表
        tokens = []
        try:
            # generate_tokens 需要一个 readline 函数
            token_gen = tokenize.generate_tokens(source_code_io.readline)
            for token_type, token_string, start, end, line_number in token_gen:
                # 检查 token 类型是否为注释
                if token_type != tokenize.COMMENT:
                    # 如果不是注释，保留 token 的类型和字符串内容
                    tokens.append((token_type, token_string))
        except tokenize.TokenError as e:
            # 捕获 token 解析错误，可能是文件语法不完整或错误
            print(f"警告: 文件 {filepath} 无法解析（TokenError: {e}），跳过。")
            return  # 跳过此文件
        # 使用 untokenize 从 token 列表重建源代码
        # untokenize 会尝试还原原始的格式（空格、换行等）
        try:
            new_source_code = tokenize.untokenize(tokens)
        except Exception as e:
            # 捕获 untokenize 过程中的其他潜在错误
            print(f"警告: 文件 {filepath} 重建源代码失败（Error: {e}），跳过。")
            return  # 跳过此文件
        # 将修改后的代码写回文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source_code)
        print(f"已处理: {filepath}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
    except IOError as e:
        print(f"错误: 读取/写入文件 {filepath} 时发生 I/O 错误: {e}")
    except Exception as e:
        # 捕获其他未知错误
        print(f"错误: 处理文件 {filepath} 时发生未知错误: {e}")


def process_directory(directory="."):
    """
    遍历指定目录及其子目录，处理所有 .py 文件。
    Args:
        directory (str): 要开始处理的目录路径，默认为当前目录。
    """
    print(f"开始清除目录 '{directory}' 及其子目录下所有 .py 文件的 # 注释...")
    # 获取当前正在运行的脚本的绝对路径，以便跳过它
    script_path = os.path.abspath(__file__)
    processed_count = 0
    skipped_count = 0
    # 遍历目录树
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # 检查文件是否以 .py 结尾
            if filename.endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                absolute_filepath = os.path.abspath(filepath)
                # 跳过脚本自身
                if absolute_filepath == script_path:
                    print(f"跳过脚本文件: {filepath}")
                    skipped_count += 1
                    continue
                # 处理 .py 文件
                remove_comments_from_file(filepath)
                processed_count += 1
    print("--- 完成 ---")
    print(f"总共找到并尝试处理 {processed_count + skipped_count} 个 .py 文件。")
    print(f"成功处理 {processed_count} 个文件。")
    print(f"跳过 {skipped_count} 个文件 (包括脚本自身)。")


if __name__ == "__main__":
    # 可以通过命令行参数指定目录，否则使用当前目录
    target_directory = "."
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
        if not os.path.isdir(target_directory):
            print(f"错误: 指定的路径 '{target_directory}' 不是一个有效的目录。")
            sys.exit(1)
    process_directory(target_directory)
