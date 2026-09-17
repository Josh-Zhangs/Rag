from fastapi import APIRouter
from services.vector_db_service import vector_db_service

# 创建 API 路由器，前缀为 "/document-stats"，标签为 "document-stats"
router = APIRouter(prefix="/document-stats", tags=["document-stats"])


@router.get("")
# 获取文档统计信息
async def get_document_stats():
    result = vector_db_service.get_document_stats()
    return {
        "code": 0,
        "msg": "success",
        "data": {
            "stats": result.get("stats", {}),
            "last_updated": result.get("last_updated", ""),
        },
    }