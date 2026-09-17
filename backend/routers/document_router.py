from fastapi import APIRouter
from services.vector_db_service import vector_db_service
from config.settings import settings

# 创建 API 路由器，前缀为 "/documents"，标签为 "documents"
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def get_documents():
    result = vector_db_service.get_documents(
        table_name=settings.lancedb_table_name,
    )
    return {
        "code": 0 if result.get("status") == "success" else 1,
        "msg": result.get("status", "unknown"),
        "data": result.get("documents", []),
    }