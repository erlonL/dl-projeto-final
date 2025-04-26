# Projeto Final de Deep Learning

Detecção e Segmentação de Semáforos de trânsito !

## Abordagem

1. Primeiro detectar onde está o objeto na imagem
2. Transformar a imagem para ficar "flat" (spatial transformer)
3. Detectar qual objeto é (verde, vermelho, amarelo)

## Motivações

1. Carros autômatos
2. Redução de escopo (apenas semáforo)
3. Semáforo é uma sinalização universal

## Dataset

- Redução do dataset do artigo br
- Vídeos do youtube
- 3 classes: verde, vermelho, amarelo
- outras fontes

## Arquitetura

- abordagem parecida com a do artigo
- transfer learning
- mask CNN

## Links

https://journals-sol.sbc.org.br/index.php/jbcs/article/view/3678/2794
https://arxiv.org/abs/1703.06870
https://github.com/xuexingyu24/License_Plate_Detection_Pytorch
Region Proposal Network (RPN)
