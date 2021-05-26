# Proyecto 4: Informe Final Carrera
## Web Scraping al portal SIATA & Correlación hidroclimática con el dengue en Medellín
___

Presentado por [David Serna Gutiérrez](https://www.linkedin.com/in/dsernag/).

Notebook disponible en el repositorio de Git-Hub: https://github.com/dsernag/Data_Science_Acamica

Ingeniero Forestal de la Universidad Nacional de Colombia Sede Medellín

Estudiante de Especialización en Sistemas de Información Geográfica

*© Todos los derechos reservados*

___

### OBJETIVO
___

El presente notebook continúa el análisis realizado en el TP3 sobre datos de [dengue](http://medata.gov.co/dataset/dengue). Los cuales corresponden al registro de pacientes atendidos en las Instituciones Prestadoras de Servicios de Salud con diagnóstico probable o confirmado de Dengue y notificados al Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA) desde el año 2008 al 2018.

Se pretende realizar web scraping al portal de descargas del "Sistema de Alerta Temprana de Medellín y el Valle de Aburrá" ([SIATA](https://siata.gov.co/descarga_siata/index.php/index2/login)). Allí hay acceso a toda la información. Cómo insumo del trabajo se descargará precipitación y temperatura. Es necesario crear una cuenta para acceder al portal, es completamente gratuito y expedito.

Luego de recolectar la información se debe realizar una depuración de los mismos, pues el portal genera 1 archivo *.csv* para cada mes de cada estación. Así, sí queremos información de 2008 a 2018, serían 11 años por 12 meses por el número de estaciones que se quieren, para 10 estaciones serían 1.320 csv. Esto igual para los datos de temperatura.

Finalmente con los datos organizados y resumidos a resolución semanal, se repetirá el diseño de features, el análisis de correlaciones y se ensayarán 3 modelos; un modelo persistente (donde el valor anterior 't-1' predecira el siguiente valor 't') como benchmark, una red neuronal LSTM univariada (únicamente con los casos de dengue) y nuevamente una red neuronal LSTM multivariada, tomando como insumos las variables hidroclimáticas. Teniendo estos tres modelos se pretende evaluar la efectividad e influencia que puedan tener las variables hidroclimáticas en la predicción de casos de dengue.
___
