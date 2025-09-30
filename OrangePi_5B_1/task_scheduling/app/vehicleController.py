import os.path
import time
import httpx
from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import threading
import numpy as np
import cv2
from utils.nacos_utils import SERVICE_IP, SERVICE_PORT
from utils import (Logger,
                   RKNNPoolExecutor,
                   ResponseModel,
                   setup_logging,
                   stitch_handler_start,
                   single_handler_start,
                   start_nacos,
                   image_pre_process,
                   decode_result,
                   Yolov8SegExecutor,
                   decode_seg,
                   yolov8m_seg_choose_service_strategy_by_circle,
                   yolov8s_seg_choose_service_strategy_by_circle,
                   yolov8m_seg_choose_service_strategy_by_weight,
                   yolov8s_seg_choose_service_strategy_by_weight,
                   start_background_updater_service_queue_length
                   )

app = FastAPI(on_startup=[setup_logging])

RKNN_MODEL = "./model/best_Pose_RepVGG_ReLU_train_opset19_deploy.rknn"
YOLOV8M_SEG = "./model/yolov8m_seg.rknn"
YOLOV8S_SEG = "./model/yolov8s_seg.rknn"

# 自定义异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        ResponseModel.error_response(
            code=exc.status_code,
            msg=exc.detail
        ).model_dump(),
        status_code=exc.status_code
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    # 生产环境建议不返回具体错误信息
    Logger.error(f"未处理异常: {str(exc)}")
    error_msg = "内部服务器错误"
    return JSONResponse(
        ResponseModel.error_response(
            code=500,
            msg=error_msg
        ).model_dump(),
        status_code=500
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })

    return JSONResponse(
        ResponseModel.error_response(
            code=400,
            msg="请求参数验证失败",
            data=errors
        ).model_dump(),
        status_code=400
    )


@app.post("/infer")
async def infer(image: UploadFile, timeout: float = Form(0.5)):
    try:
        # 验证文件类型
        if not image.content_type.startswith("image/"):
            raise HTTPException(400, "仅支持图片文件")

        # 异步读取文件
        img_bytes = await image.read()
        # 验证文件大小
        if len(img_bytes) > 5 * 1024 * 1024:
            raise HTTPException(413, "图片大小超过5MB限制")
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(400, "无效的图片格式")

        inferImage = image_pre_process(img)

        # 提交推理
        future = executor.submit(inferImage)

        try:
            model_outputs = future.result(timeout=timeout)

            result = decode_result(model_outputs, inferImage)

            # 使用 JSONResponse 包装响应
            return JSONResponse(
                content=ResponseModel.success_response(data=result).model_dump(),
                status_code=200
            )
        except TimeoutError:
            raise HTTPException(408, f"推理超时(>{timeout}秒)")
        except Exception as e:
            raise HTTPException(500, f"推理失败: {str(e)}")
    except HTTPException as e:
        raise e  # 让异常处理器处理
    except Exception as e:
        raise HTTPException(500, f"服务器内部错误: {str(e)}")


@app.post("/imageSeg/yolov8m")
async def imageSeg_yolov8m(image: UploadFile, timeout: float = Form(2.0), service_type: str = Form("local")):
    # Logger.success(f"timeout: {timeout}, service_type: {service_type}")

    detect_flag = False
    target_url = ""

    service = None

    if service_type == "local":
        detect_flag = True
    elif service_type == "circle":
        service = yolov8m_seg_choose_service_strategy_by_circle()
        if service is None:
            detect_flag = True
        else:
            if service.host == SERVICE_IP and str(service.port) == str(SERVICE_PORT):
                detect_flag = True
            else:
                target_url = f"http://{service.host}:{service.port}/imageSeg/yolov8m"
    elif service_type == "weight":
        service = yolov8m_seg_choose_service_strategy_by_weight()
        if service is None:
            detect_flag = True
        else:
            if service.host == SERVICE_IP and str(service.port) == str(SERVICE_PORT):
                detect_flag = True
            else:
                target_url = f"http://{service.host}:{service.port}/imageSeg/yolov8m"
    else:
        detect_flag = True

    start_time = time.time()

    if detect_flag:
        try:
            # 验证文件类型
            if not image.content_type.startswith("image/"):
                raise HTTPException(400, "仅支持图片文件")

            # 异步读取文件
            img_bytes = await image.read()
            # 验证文件大小
            if len(img_bytes) > 5 * 1024 * 1024:
                raise HTTPException(413, "图片大小超过5MB限制")
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(400, "无效的图片格式")

            if service is not None:
                service.task_cost += 4.0

            inferImage = image_pre_process(img)

            # 提交推理
            future = yolov8m_seg_executor.submit(inferImage)

            try:
                model_outputs = future.result(timeout=timeout)

                result = decode_seg(model_outputs, inferImage)

                if service is not None:
                    service.task_cost -= 4.0

                if service_type != "local" and service is not None:
                    service.yolov8m_seg_response_time_list.append(time.time() - start_time)

                # 使用 JSONResponse 包装响应
                return JSONResponse(
                    content=ResponseModel.success_response(data=result).model_dump(),
                    status_code=200
                )
            except TimeoutError:
                raise HTTPException(408, f"推理超时(>{timeout}秒)")
            except Exception as e:
                raise HTTPException(500, f"推理失败: {str(e)}")
        except HTTPException as e:
            raise e  # 让异常处理器处理
        except Exception as e:
            raise HTTPException(500, f"服务器内部错误: {str(e)}")
    else:
        try:
            # 构造 multipart/form-data 请求体
            files = {"image": (image.filename, await image.read(), image.content_type)}
            data = {
                "timeout": timeout,
                "service_type": "local"
            }

            if service is not None:
                service.task_cost += 4.0

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    target_url,
                    files=files,
                    data=data,
                    timeout=timeout
                )

                if service is not None:
                    service.task_cost -= 4.0

                if service_type != "local" and service is not None:
                    service.yolov8m_seg_response_time_list.append(time.time() - start_time)

                # 直接返回目标服务的响应内容和状态码
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code
                )

        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail=f"目标服务响应超时(>{timeout}秒)")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"目标服务不可用: {str(e)}")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")



@app.post("/imageSeg/yolov8s")
async def imageSeg_yolov8s(image: UploadFile, timeout: float = Form(2.0), service_type: str = Form("local")):
    # Logger.success(f"timeout: {timeout}, service_type: {service_type}")

    detect_flag = False
    target_url = ""

    service = None

    if service_type == "local":
        detect_flag = True
    elif service_type == "circle":
        service = yolov8s_seg_choose_service_strategy_by_circle()
        if service is None:
            detect_flag = True
        else:
            if service.host == SERVICE_IP and str(service.port) == str(SERVICE_PORT):
                detect_flag = True
            else:
                target_url = f"http://{service.host}:{service.port}/imageSeg/yolov8s"
    elif service_type == "weight":
        service = yolov8s_seg_choose_service_strategy_by_weight()
        if service is None:
            detect_flag = True
        else:
            if service.host == SERVICE_IP and str(service.port) == str(SERVICE_PORT):
                detect_flag = True
            else:
                target_url = f"http://{service.host}:{service.port}/imageSeg/yolov8s"
    else:
        detect_flag = True

    start_time = time.time()

    if detect_flag:
        try:
            # 验证文件类型
            if not image.content_type.startswith("image/"):
                raise HTTPException(400, "仅支持图片文件")

            # 异步读取文件
            img_bytes = await image.read()
            # 验证文件大小
            if len(img_bytes) > 5 * 1024 * 1024:
                raise HTTPException(413, "图片大小超过5MB限制")
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(400, "无效的图片格式")

            if service is not None:
                service.task_cost += 2.0

            inferImage = image_pre_process(img)

            # 提交推理
            future = yolov8s_seg_executor.submit(inferImage)

            try:
                model_outputs = future.result(timeout=timeout)

                result = decode_seg(model_outputs, inferImage)

                if service is not None:
                    service.task_cost -= 2.0

                if service_type != "local" and service is not None:
                    service.yolov8s_seg_response_time_list.append(time.time() - start_time)

                # 使用 JSONResponse 包装响应
                return JSONResponse(
                    content=ResponseModel.success_response(data=result).model_dump(),
                    status_code=200
                )
            except TimeoutError:
                raise HTTPException(408, f"推理超时(>{timeout}秒)")
            except Exception as e:
                raise HTTPException(500, f"推理失败: {str(e)}")
        except HTTPException as e:
            raise e  # 让异常处理器处理
        except Exception as e:
            raise HTTPException(500, f"服务器内部错误: {str(e)}")
    else:
        # 构造 multipart/form-data 请求体
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {
            "timeout": timeout,
            "service_type": "local"
        }

        if service is not None:
            service.task_cost += 2.0

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    target_url,
                    files=files,
                    data=data,
                    timeout=timeout
                )

                if service is not None:
                    service.task_cost -= 2.0

                if service_type != "local" and service is not None:
                    service.yolov8s_seg_response_time_list.append(time.time() - start_time)

                # 直接返回目标服务的响应内容和状态码
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code
                )

        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail=f"目标服务响应超时(>{timeout}秒)")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"目标服务不可用: {str(e)}")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


@app.get("/queueLength")
async def queueLength():
    yolov8n_pose_queue_length = executor.task_queue.qsize()
    yolov8s_seg_queue_length = yolov8s_seg_executor.task_queue.qsize()
    yolov8m_seg_queue_length = yolov8m_seg_executor.task_queue.qsize()

    # 构造响应数据
    response_data = {
        "yolov8n_pose_queue_length": yolov8n_pose_queue_length,
        "yolov8s_seg_queue_length": yolov8s_seg_queue_length,
        "yolov8m_seg_queue_length": yolov8m_seg_queue_length
    }

    # 返回 JSON 格式的响应
    return JSONResponse(content=response_data)


if __name__ == '__main__':

    sava_prefix = "./result/weight_ffmpeg_7/stitch_handler_weight_response_2"
    SAMPLE_INTERVAL = 1.0
    THREAD_SAMPLE_INTERVAL = 0.2

    metrics_prefix = os.path.join(sava_prefix, "metrics")       # 存储各项指标
    response_time_prefix = os.path.join(sava_prefix, "response_time")       # 存储接口响应时间
    image_prefix = os.path.join(sava_prefix, "image")       # 存储推理图片
    frame_rate_prefix = os.path.join(sava_prefix, "frame_rate")     # 存储帧处理速率
    image_time_prefix = os.path.join(sava_prefix, "get_image_time")     # 存储获取图片速率
    frame_handler_time_prefix = os.path.join(sava_prefix, "frame_handler_time")     # 存储处理一帧的平均耗时

    # 创建执行器
    executor = RKNNPoolExecutor(rknn_model=RKNN_MODEL, max_workers=3)

    yolov8m_seg_executor = Yolov8SegExecutor(rknn_model=YOLOV8M_SEG, max_workers=3)
    yolov8s_seg_executor = Yolov8SegExecutor(rknn_model=YOLOV8S_SEG, max_workers=3)

    # 启动nacos，并进行配置更新
    threading.Thread(target=start_nacos, args=(metrics_prefix,), daemon=True).start()

    start_background_updater_service_queue_length(0.5)

    # 启动视频处理监听线程
    # threading.Thread(target=single_handler_start, args=(20, SAMPLE_INTERVAL, THREAD_SAMPLE_INTERVAL, response_time_prefix, image_prefix, frame_rate_prefix, image_time_prefix, frame_handler_time_prefix, ), daemon=True).start()
    threading.Thread(target=stitch_handler_start, args=(20, SAMPLE_INTERVAL, THREAD_SAMPLE_INTERVAL, response_time_prefix, image_prefix, frame_rate_prefix, image_time_prefix, frame_handler_time_prefix, ), daemon=True).start()

    uvicorn.run(app, host='0.0.0.0', port=8080, log_config=None)

