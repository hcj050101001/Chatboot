"""按 .env 配置自动创建 ERP 演示数据库和样例数据。"""
import os
import re
from pathlib import Path

import pymysql
from dotenv import load_dotenv

BASE_DIR=Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SCHEMA_STATEMENTS=(
    """CREATE TABLE IF NOT EXISTS suppliers (
        supplier_id VARCHAR(20) PRIMARY KEY,
        supplier_name VARCHAR(100) NOT NULL UNIQUE,
        contact_name VARCHAR(50), phone VARCHAR(20), email VARCHAR(100),
        status VARCHAR(20) NOT NULL DEFAULT '启用'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS products (
        product_id VARCHAR(20) PRIMARY KEY,
        product_name VARCHAR(100) NOT NULL UNIQUE,
        category VARCHAR(50), unit VARCHAR(20) NOT NULL DEFAULT '件',
        current_stock DECIMAL(12,2) NOT NULL DEFAULT 0,
        safety_stock DECIMAL(12,2) NOT NULL DEFAULT 0,
        unit_price DECIMAL(12,2) NOT NULL DEFAULT 0,
        status VARCHAR(20) NOT NULL DEFAULT '在售'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS purchase_orders (
        purchase_order_id VARCHAR(30) PRIMARY KEY,
        supplier_id VARCHAR(20) NOT NULL,
        order_date DATE NOT NULL, expected_arrival_date DATE,
        actual_arrival_date DATE, order_status VARCHAR(20) NOT NULL,
        total_amount DECIMAL(14,2) NOT NULL DEFAULT 0,
        CONSTRAINT fk_po_supplier FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS purchase_order_items (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        purchase_order_id VARCHAR(30) NOT NULL, product_id VARCHAR(20) NOT NULL,
        quantity DECIMAL(12,2) NOT NULL, received_quantity DECIMAL(12,2) NOT NULL DEFAULT 0,
        unit_price DECIMAL(12,2) NOT NULL,
        UNIQUE KEY uk_purchase_item (purchase_order_id, product_id),
        CONSTRAINT fk_poi_order FOREIGN KEY (purchase_order_id) REFERENCES purchase_orders(purchase_order_id),
        CONSTRAINT fk_poi_product FOREIGN KEY (product_id) REFERENCES products(product_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS customers (
        customer_id VARCHAR(20) PRIMARY KEY,
        customer_name VARCHAR(100) NOT NULL UNIQUE,
        contact_name VARCHAR(50), phone VARCHAR(20),
        status VARCHAR(20) NOT NULL DEFAULT '合作中'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS sales_orders (
        sales_order_id VARCHAR(30) PRIMARY KEY,
        customer_id VARCHAR(20) NOT NULL, order_date DATE NOT NULL,
        order_status VARCHAR(20) NOT NULL, total_amount DECIMAL(14,2) NOT NULL DEFAULT 0,
        CONSTRAINT fk_so_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS sales_order_items (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        sales_order_id VARCHAR(30) NOT NULL, product_id VARCHAR(20) NOT NULL,
        quantity DECIMAL(12,2) NOT NULL, unit_price DECIMAL(12,2) NOT NULL,
        UNIQUE KEY uk_sales_item (sales_order_id, product_id),
        CONSTRAINT fk_soi_order FOREIGN KEY (sales_order_id) REFERENCES sales_orders(sales_order_id),
        CONSTRAINT fk_soi_product FOREIGN KEY (product_id) REFERENCES products(product_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS production_orders (
        production_order_id VARCHAR(30) PRIMARY KEY,
        product_id VARCHAR(20) NOT NULL,
        planned_quantity DECIMAL(12,2) NOT NULL,
        completed_quantity DECIMAL(12,2) NOT NULL DEFAULT 0,
        start_date DATE, due_date DATE, production_status VARCHAR(20) NOT NULL,
        CONSTRAINT fk_production_product FOREIGN KEY (product_id) REFERENCES products(product_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
)

SEED_STATEMENTS=(
    ("INSERT IGNORE INTO suppliers VALUES (%s,%s,%s,%s,%s,%s)",(
        ("SUP001","A供应商","陈采购","13900000001","contact_a@example.com","启用"),
        ("SUP002","B供应商","周采购","13900000002","contact_b@example.com","启用"),
        ("SUP003","C供应商","吴采购","13900000003","contact_c@example.com","启用"),
    )),
    ("INSERT IGNORE INTO products VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",(
        ("PRD001","工业控制器","电子元件","台",8,20,1200,"在售"),
        ("PRD002","精密轴承","机械配件","套",45,80,260,"在售"),
        ("PRD003","传动齿轮","机械配件","件",150,100,180,"在售"),
        ("PRD004","工业传感器","电子元件","个",12,30,450,"在售"),
    )),
    ("INSERT IGNORE INTO purchase_orders VALUES (%s,%s,%s,%s,%s,%s,%s)",(
        ("PO202606001","SUP002","2026-06-20","2026-06-28","2026-06-27","已到货",26000),
        ("PO202607001","SUP001","2026-07-01","2026-07-10",None,"待到货",39600),
        ("PO202607002","SUP001","2026-07-03","2026-07-12",None,"部分到货",18000),
    )),
    ("INSERT IGNORE INTO purchase_order_items (id,purchase_order_id,product_id,quantity,received_quantity,unit_price) VALUES (%s,%s,%s,%s,%s,%s)",(
        (5,"PO202607001","PRD001",20,0,1200),(6,"PO202607001","PRD002",60,0,260),
        (7,"PO202607002","PRD004",40,10,450),(8,"PO202606001","PRD002",100,100,260),
    )),
    ("INSERT IGNORE INTO customers VALUES (%s,%s,%s,%s,%s)",(
        ("CUST001","华东制造有限公司","刘经理","13700000001","合作中"),
        ("CUST002","华北科工有限公司","赵经理","13700000002","合作中"),
        ("CUST003","华南设备有限公司","孙经理","13700000003","合作中"),
    )),
    ("INSERT IGNORE INTO sales_orders VALUES (%s,%s,%s,%s,%s)",(
        ("SO202607001","CUST001","2026-07-02","已完成",32400),
        ("SO202607002","CUST001","2026-07-08","已完成",9000),
        ("SO202607003","CUST002","2026-07-05","已完成",18000),
        ("SO202607004","CUST003","2026-07-09","已完成",12600),
    )),
    ("INSERT IGNORE INTO sales_order_items (id,sales_order_id,product_id,quantity,unit_price) VALUES (%s,%s,%s,%s,%s)",(
        (1,"SO202607001","PRD001",20,1200),(2,"SO202607001","PRD002",30,280),
        (3,"SO202607002","PRD004",20,450),(4,"SO202607003","PRD003",100,180),
        (5,"SO202607004","PRD002",45,280),
    )),
    ("INSERT IGNORE INTO production_orders VALUES (%s,%s,%s,%s,%s,%s,%s)",(
        ("MO202607001","PRD001",50,20,"2026-07-04","2026-07-15","生产中"),
        ("MO202607002","PRD003",200,200,"2026-07-01","2026-07-08","已完成"),
    )),
)


def initialize_database() -> bool:
    """创建数据库、数据表和演示数据；失败时返回 False，不阻止网页启动。"""
    database=os.getenv("DB_NAME","erp_db").strip()
    if not re.fullmatch(r"[A-Za-z0-9_]+",database):
        print("数据库初始化失败：DB_NAME 只能包含字母、数字和下划线")
        return False
    user=os.getenv("DB_USER","").strip()
    password=os.getenv("DB_PASSWORD","")
    if not user or not password or password.startswith("请填写"):
        print("未配置可用的 MySQL 用户名/密码，跳过 ERP 数据库初始化。")
        return False

    connection=None
    try:
        connection=pymysql.connect(
            host=os.getenv("DB_HOST","localhost"),port=int(os.getenv("DB_PORT","3306")),
            user=user,password=password,charset="utf8mb4",autocommit=False,
        )
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` DEFAULT CHARACTER SET utf8mb4")
            cursor.execute(f"USE `{database}`")
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
            for statement,rows in SEED_STATEMENTS:
                cursor.executemany(statement,rows)
        connection.commit()
        print(f"ERP 演示数据库已就绪：{database}")
        return True
    except Exception as error:
        if connection:
            connection.rollback()
        print(f"ERP 数据库初始化失败：{error}")
        return False
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(0 if initialize_database() else 1)
