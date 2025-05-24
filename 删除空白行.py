import os


def remove_empty_lines_from_file(filepath):
    """
    从指定文件中删除所有仅包含空白字符的行。
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) < len(lines):
            with open(filepath, "w", encoding="utf-8") as file:
                file.writelines(non_empty_lines)
            print(f"已处理并修改: {filepath}")
            return True
        else:
            return False
    except FileNotFoundError:
        print(f"错误: 文件未找到 {filepath}")
        return False
    except Exception as e:
        print(f"处理文件 {filepath} 时发生错误: {e}")
        return False


def process_directory(directory="."):
    """
    递归处理指定目录及其子目录下的所有 .py 文件。
    """
    modified_count = 0
    processed_count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                processed_count += 1
                if remove_empty_lines_from_file(filepath):
                    modified_count += 1
    print(
        f"\n处理完成。共检查 {processed_count} 个 .py 文件，其中 {modified_count} 个文件被修改。"
    )


if __name__ == "__main__":
    confirm = input(
        "警告：此操作将直接修改当前目录及其子目录下所有 .py 文件，删除仅包含空白字符的行。\n"
        "建议在执行前备份您的代码。\n"
        "是否继续? (yes/no): "
    )
    if confirm.lower() == "yes":
        print("开始处理...")
        process_directory(".")
    else:
        print("操作已取消。")
