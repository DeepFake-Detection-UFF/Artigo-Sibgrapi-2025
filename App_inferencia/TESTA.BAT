rem Se você estiver usando EfficientNet-B7, a camada layer4 não existe. Use algo como:

rem --target_layer "features.6" ou --target_layer "features.7"
rem Se o Grad-CAM passar mas o mapa ficar “apagado”, teste outra features.* mais profunda.

REM python3 app_inferencia.py --model E:\MODELS\modelo_resnet50_frames.pth --video  "E:\videos\video1.mp4" --frames 5 --segment_face  --outdir E:\models_outputs 


rem python3 app_inferencia.py --model E:\MODELS\modelo_efficientnetb7_frames.pth --video  "E:\videos\video1.mp4" --frames 5 --segment_face  --outdir E:\models_outputs 


rem python3 app_inferencia.py --model E:\MODELS\modelo_efficientnetb7_frames.pth --video  "E:\videos\video3.mp4" --frames 5 --segment_face  --outdir E:\models_outputs  --enable_smoothgrad

python3 app_inferencia.py ^
  --model "E:\MODELS\modelo_efficientnetb7_frames.pth" ^
  --video "E:\videos\video3.mp4" --frames 5 --segment_face ^
  --outdir "E:\models_outputs" --enable_smoothgrad ^
  --target_layer "layer4"
