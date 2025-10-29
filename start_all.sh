#!/bin/bash

echo "========================================="
echo "启动所有服务"
echo "========================================="

# 停止所有服务
echo "1. 停止现有服务..."
pkill -9 -f "start_backend"
pkill -9 -f "vite"
sleep 2

# 启动后端
echo "2. 启动后端 (端口 8000)..."
cd ./backend
nohup python start_backend.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "   后端PID: $BACKEND_PID"

# 等待后端启动（PaddleOCR模型加载需要时间）
echo "3. 等待后端启动（加载PaddleOCR模型，需要15-20秒）..."
sleep 15

# 检查后端
echo "4. 测试后端..."
HEALTH=$(curl -s http://localhost:8000/ | grep "healthy")
if [ -n "$HEALTH" ]; then
    echo "   ✅ 后端运行正常"
else
    echo "   ❌ 后端启动失败"
    tail -20 backend.log
    exit 1
fi

# 启动前端
echo "5. 启动前端 (端口 3001)..."
cd ../frontend
PORT=3001 nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   前端PID: $FRONTEND_PID"

# 等待前端启动
echo "6. 等待前端启动..."
sleep 5

echo ""
echo "========================================="
echo "✅ 服务启动完成"
echo "========================================="
echo "前端地址: http://192.168.110.131:3001/"
echo "后端地址: http://192.168.110.131:8000/"

