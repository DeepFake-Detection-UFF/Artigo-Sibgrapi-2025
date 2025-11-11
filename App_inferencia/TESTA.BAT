rem Se você estiver usando EfficientNet-B7, a camada layer4 não existe. Use algo como:

rem --target_layer "features.6" ou --target_layer "features.7"
rem Se o Grad-CAM passar mas o mapa ficar “apagado”, teste outra features.* mais profunda.

REM python3 app_inferencia.py --model E:\MODELS\modelo_resnet50_frames.pth --video  "E:\videos\video1.mp4" --frames 5 --segment_face  --outdir E:\models_outputs 


rem python3 app_inferencia.py --model E:\MODELS\modelo_efficientnetb7_frames.pth --video  "E:\videos\video1.mp4" --frames 5 --segment_face  --outdir E:\models_outputs 


rem python3 app_inferencia.py --model E:\MODELS\modelo_efficientnetb7_frames.pth --video  "E:\videos\video3.mp4" --frames 5 --segment_face  --outdir E:\models_outputs  --enable_smoothgrad

rem python3 app_inferencia.py ^
rem   --model "E:\MODELS\modelo_efficientnetb7_frames.pth" ^
rem   --video "E:\videos\video3.mp4" --frames 5 --segment_face ^
rem   --outdir "E:\models_outputs" --enable_smoothgrad ^
rem   --target_layer "layer4"


rem python3 "E:\bc_niteroi_rj\itau\app_inferencia\app_inferencia.py" ^
rem   --model "E:\MODELS\modelo_efficientnetb7_frames.pth" ^
rem   --image "E:\faceforensis_face_maskgan\test_224\manipulated_mouth_224\original_008_frame4_manipulated_mouth.png" ^
rem   --segment_face ^
rem   --outdir "E:\models_outputs" ^
rem   --enable_smoothgrad ^
rem   --target_layer "features.8"


python3 "E:\bc_niteroi_rj\itau\app_inferencia\app_inferencia.py" ^
  --model "E:\MODELS\modelo_resnet50_frames.pth " ^
  --image "E:\faceforensis_face_maskgan\test_224\manipulated_mouth_224\original_008_frame4_manipulated_mouth.png" ^
  --segment_face ^
  --outdir "E:\models_outputs" ^
  --enable_smoothgrad ^
  --target_layer "layer4"

