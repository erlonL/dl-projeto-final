# Projeto Final de Deep Learning

## 🚦 Detecção e Classificação de Semáforos de trânsito

## Introdução

Projeto Final da Disciplina de Deep Learning

Veja o [vídeo do projeto](https://youtu.be/7b31YjWQJwU)

---

## Organização das pastas

O Código-fonte principal do projeto está disponível na pasta `src/`  
e está dividido em 5 notebooks principais:

1. treat-visualize-classes.ipynb
2. train-cnn.ipynb
3. model_utils.py
4. run_detection.py
5. visualize-results.ipynb

## Outras Informações (Planejamento Inicial)

### Abordagem

1. Primeiro detectar onde está o objeto na imagem
2. Transformar a imagem para ficar "flat" (spatial transformer)
3. Detectar qual objeto é (verde, vermelho, amarelo)

### Motivações

1. Carros autômatos
2. Redução de escopo (apenas semáforo)
3. Semáforo é uma sinalização universal

### Dataset Utilizado / Outros dados

- [Bosch Small Traffic Lights Dataset](https://zenodo.org/records/12706046)
- 4 classes: 🟢verde, 🔴vermelho, 🟡amarelo, ⚫desligado
- Vídeos do youtube
- outras fontes

### Arquitetura

-> abordagem parecida com a do [artigo](https://pure.port.ac.uk/ws/portalfiles/portal/75106637/Robust_Real_time_Traffic_Light_Detector_on_Small_Form_Platform_for_Autonomous_Vehicleslastedition.pdf)

- (talvez) transfer learning para o modelo de detecção
- (talvez) mask CNN

### Links

https://journals-sol.sbc.org.br/index.php/jbcs/article/view/3678/2794  
https://arxiv.org/abs/1703.06870  
https://github.com/xuexingyu24/License_Plate_Detection_Pytorch  
Region Proposal Network (RPN)  
Paper do ResNet / Faster R-CNN
