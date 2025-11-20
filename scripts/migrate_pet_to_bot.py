"""
数据库字段迁移脚本：pet_name → bot_name

用于将 relationship_status 集合中的字段名从 pet_name 改为 bot_name
"""
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_relationship_status():
    """迁移 relationship_status 集合的字段名"""
    # 从配置文件读取连接字符串和数据库名
    # 这里使用默认值，实际使用时请从config读取
    connection_string = "mongodb://localhost:27017/"
    database_name = "bot_chat_db"  # 根据您的实际数据库名修改
    
    client = MongoClient(connection_string)
    db = client[database_name]
    collection = db["relationship_status"]
    
    try:
        # 统计需要更新的文档数量
        count = collection.count_documents({"pet_name": {"$exists": True}})
        logger.info(f"找到 {count} 个包含 pet_name 字段的文档")
        
        if count == 0:
            logger.info("没有需要迁移的数据")
            return
        
        # 重命名字段：pet_name → bot_name
        result = collection.update_many(
            {"pet_name": {"$exists": True}},
            {"$rename": {"pet_name": "bot_name"}}
        )
        
        logger.info(f"成功更新 {result.modified_count} 个文档")
        logger.info("字段迁移完成！")
        
    except Exception as e:
        logger.error(f"迁移过程中发生错误: {e}", exc_info=True)
    finally:
        client.close()

if __name__ == "__main__":
    print("=" * 60)
    print("数据库字段迁移：pet_name → bot_name")
    print("=" * 60)
    print()
    print("此脚本将更新 relationship_status 集合中的字段名")
    print("将 'pet_name' 重命名为 'bot_name'")
    print()
    
    confirm = input("确认执行迁移? (yes/no): ")
    if confirm.lower() == 'yes':
        migrate_relationship_status()
    else:
        print("迁移已取消")
