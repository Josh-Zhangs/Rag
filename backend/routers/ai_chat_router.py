from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.chat import AIChatRequest
from services.ai_chat_service import ai_chat_service

router = APIRouter(prefix='/ai-chat', tags=['ai-chat'])

@router.post('/stream')
async def chat_stream(request: AIChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail='質問を入力してください')

    # 将 Pydantic ConversationTurn 列表转为普通 dict 列表
    history = [{"question": t.question, "answer": t.answer} for t in request.history]
    # 返回 StreamingResponse，media_type 为 text/event-stream
    # process_question_stream 是一个生成器，每次 yield 一条 SSE 消息
    return StreamingResponse(
        ai_chat_service.process_question_stream(question, history),
        media_type="text/event-stream; charset=utf-8",
        headers={
            # 禁用缓存，确保前端实时接收
            "Cache-Control": "no-cache",
            # 禁用代理缓冲（nginx 等反向代理场景）
            "X-Accel-Buffering": "no",
            # 保持连接，兼容 SSE 协议
            "Connection": "keep-alive",
        },
    )