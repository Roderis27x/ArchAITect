import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import os
import gradio as gr
import logging
import datetime

# Configurar logging para mostrar solo mensajes de WARNING y superiores
logging.basicConfig(level=logging.WARNING)

# Configuración para suprimir advertencias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Obtener y mostrar el directorio actual
current_dir = os.getcwd()
print(f"Directorio actual: {current_dir}")

# Verificar si el archivo existe
modelo_path = 'ArchiNet.h5'
if os.path.exists(modelo_path):
    print(f"El modelo se encontró en: {os.path.abspath(modelo_path)}")
else:
    print(f"ERROR: No se encuentra el archivo {modelo_path}")
    print(f"Buscando en: {os.path.abspath(modelo_path)}")

# Diccionario con información detallada de cada estilo
estilos_info = {
    'Arquitectura aquemenida': {
        'epoca': 'Imperio Persa (550-330 a.C.)',
        'caracteristicas': 'Columnas altas y delgadas, terrazas monumentales, escaleras simétricas y relieves decorativos',
        'materiales': 'Piedra caliza, mármol y madera de cedro',
        'relacion': 'Influenciada por la arquitectura mesopotámica y egipcia'
    },
    'Estilo artesano americano': {
        'epoca': 'Finales del siglo XIX - principios del siglo XX',
        'caracteristicas': 'Porches amplios, vigas expuestas, líneas horizontales y trabajo artesanal detallado',
        'materiales': 'Madera local, piedra y ladrillo',
        'relacion': 'Reacción contra la era industrial y el estilo victoriano'
    },
    'Arquitectura American Foursquare': {
        'epoca': '1890s-1930s',
        'caracteristicas': 'Forma cúbica, dos pisos y medio, techo en pirámide, porche frontal ancho',
        'materiales': 'Madera, ladrillo o piedra arenisca',
        'relacion': 'Derivado del estilo Prairie School y una reacción al ornamentado estilo victoriano'
    },
    'Arquitectura del antiguo Egipto': {
        'epoca': '3000 a.C. - 30 a.C.',
        'caracteristicas': 'Monumentalidad, simetría, columnas masivas, jeroglíficos y relieves',
        'materiales': 'Piedra caliza, granito y arenisca',
        'relacion': 'Influyó en muchos estilos posteriores, incluyendo el neoclásico'
    },
    'Arquitectura Art Deco': {
        'epoca': '1920s-1930s',
        'caracteristicas': 'Formas geométricas, zigzags, motivos escalonados y decoración suntuosa',
        'materiales': 'Hormigón, acero, vidrio y materiales lujosos como el mármol',
        'relacion': 'Evolución del Art Nouveau y precursor del estilo internacional'
    },
    'Arquitectura Art Nouveau': {
        'epoca': '1890-1910',
        'caracteristicas': 'Líneas curvas orgánicas, motivos florales y naturales, asimetría',
        'materiales': 'Hierro forjado, vidrio, cerámica y madera',
        'relacion': 'Reacción contra el academicismo del siglo XIX'
    },
    'Arquitectura barroca': {
        'epoca': 'Siglos XVII-XVIII',
        'caracteristicas': 'Dramatismo, ornamentación excesiva, curvas y contracurvas',
        'materiales': 'Mármol, piedra, estuco y oro',
        'relacion': 'Evolución del Renacimiento y precedente del Rococó'
    },
    'Arquitectura Bauhaus': {
        'epoca': '1919-1933',
        'caracteristicas': 'Funcionalismo, simplicidad, geometría pura, ausencia de ornamentación',
        'materiales': 'Acero, hormigón, vidrio',
        'relacion': 'Influenció enormemente el Movimiento Moderno'
    },
    'Arquitectura Beaux-Arts': {
        'epoca': '1880s-1920s',
        'caracteristicas': 'Simetría, grandiosidad, ornamentación clásica elaborada',
        'materiales': 'Piedra, mármol, hierro forjado',
        'relacion': 'Basado en los principios clásicos de la arquitectura grecorromana'
    },
    'Arquitectura bizantina': {
        'epoca': 'Siglos VI-XV',
        'caracteristicas': 'Cúpulas sobre planta cuadrada, mosaicos, arcos de medio punto',
        'materiales': 'Ladrillo, piedra, mosaicos y mármol',
        'relacion': 'Fusión de tradiciones romanas y orientales'
    },
    'Arquitectura de la escuela de Chicago': {
        'epoca': '1880s-1900s',
        'caracteristicas': 'Estructuras altas, ventanas grandes, énfasis en la verticalidad',
        'materiales': 'Acero, vidrio, terracota',
        'relacion': 'Precursora del modernismo y los rascacielos'
    },
    'Arquitectura colonial': {
        'epoca': 'Siglos XVII-XIX',
        'caracteristicas': 'Simetría, frontones triangulares, columnas clásicas',
        'materiales': 'Madera, ladrillo, piedra local',
        'relacion': 'Adaptación de estilos europeos a las Américas'
    },
    'Deconstructivismo': {
        'epoca': '1980s-presente',
        'caracteristicas': 'Formas fragmentadas, ángulos inusuales, aparente caos controlado',
        'materiales': 'Acero, vidrio, titanio, materiales compuestos',
        'relacion': 'Reacción contra el racionalismo del Movimiento Moderno'
    },
    'Arquitectura eduardiana': {
        'epoca': '1901-1910',
        'caracteristicas': 'Elegancia, ornamentación moderada, espacios luminosos',
        'materiales': 'Ladrillo rojo, piedra, madera',
        'relacion': 'Evolución del estilo victoriano'
    },
    'Arquitectura georgiana': {
        'epoca': '1714-1830',
        'caracteristicas': 'Simetría, proporciones clásicas, elegancia sobria',
        'materiales': 'Ladrillo, piedra, madera pintada',
        'relacion': 'Basada en el Palladianismo y el clasicismo'
    },
    'Arquitectura gótica': {
        'epoca': 'Siglos XII-XVI',
        'caracteristicas': 'Arcos apuntados, bóvedas de crucería, vidrieras, verticalidad',
        'materiales': 'Piedra, vidrio coloreado',
        'relacion': 'Evolución del románico'
    },
    'Arquitectura neogriega': {
        'epoca': '1820s-1860s',
        'caracteristicas': 'Columnas clásicas, frontones, proporciones armoniosas',
        'materiales': 'Mármol, piedra, estuco',
        'relacion': 'Revivalismo de la arquitectura de la antigua Grecia'
    },
    'Estilo internacional': {
        'epoca': '1920s-1960s',
        'caracteristicas': 'Simplicidad, funcionalidad, ausencia de ornamentación',
        'materiales': 'Acero, hormigón, vidrio',
        'relacion': 'Desarrollo del Movimiento Moderno'
    },
    'Arquitectura novelty': {
        'epoca': '1920s-1950s',
        'caracteristicas': 'Formas miméticas, diseños llamativos y temáticos',
        'materiales': 'Variados, según el diseño específico',
        'relacion': 'Relacionada con la cultura pop y el comercio'
    },
    'Arquitectura palladiana': {
        'epoca': 'Siglo XVI-XVIII',
        'caracteristicas': 'Simetría perfecta, proporciones matemáticas, pórticos con columnas',
        'materiales': 'Piedra, ladrillo, estuco',
        'relacion': 'Basada en la arquitectura clásica romana'
    },
    'Arquitectura posmoderna': {
        'epoca': '1960s-presente',
        'caracteristicas': 'Eclecticismo, ironía, referencias históricas',
        'materiales': 'Diversos, incluyendo materiales tradicionales y nuevos',
        'relacion': 'Reacción contra el Movimiento Moderno'
    },
    'Arquitectura Queen Anne': {
        'epoca': '1880s-1900s',
        'caracteristicas': 'Asimetría, torres, texturas variadas, ornamentación elaborada',
        'materiales': 'Madera, ladrillo, piedra',
        'relacion': 'Parte del movimiento victoriano'
    },
    'Arquitectura románica': {
        'epoca': 'Siglos XI-XIII',
        'caracteristicas': 'Arcos de medio punto, muros gruesos, bóvedas de cañón',
        'materiales': 'Piedra, ladrillo',
        'relacion': 'Precedente del gótico'
    },
    'Arquitectura neorrusa': {
        'epoca': '1850s-1900s',
        'caracteristicas': 'Cúpulas bulbosas, decoración colorida, formas tradicionales rusas',
        'materiales': 'Ladrillo, madera, azulejos decorativos',
        'relacion': 'Revivalismo de la arquitectura tradicional rusa'
    },
    'Arquitectura neotudor': {
        'epoca': 'Mediados del siglo XIX-presente',
        'caracteristicas': 'Entramado de madera falso, tejados empinados, chimeneas ornamentadas',
        'materiales': 'Ladrillo, estuco, madera',
        'relacion': 'Revivalismo del estilo Tudor original'
    }
}

# Lista de estilos arquitectónicos (mantener el orden original)
styles = list(estilos_info.keys())

def preprocesar_imagen(imagen):
    # Convertir imagen de RGB a BGR
    img = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalizar
    img = img.astype('float32') / 255.0
    return img

# Cargar el modelo con custom_objects
modelo_path = 'ArchiNet.h5'
if not os.path.exists(modelo_path):
    print(f"Error: No se encuentra el archivo del modelo en {modelo_path}")
    exit(1)

try:
    modelo = load_model(modelo_path, compile=False)
    # Recompilar el modelo con configuraciones específicas
    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    exit(1)

def predecir_estilo(imagen):
    try:
        img = preprocesar_imagen(imagen)
        img = np.expand_dims(img, axis=0)
        
        prediccion = modelo.predict(img, verbose=0)
        indice_predicho = np.argmax(prediccion)
        probabilidad = prediccion[0][indice_predicho]
        estilo_predicho = styles[indice_predicho]
        
        return estilo_predicho, probabilidad
    except Exception as e:
        print(f"Error al procesar la imagen: {str(e)}")
        return None, None

def predecir_estilo_gradio(imagen):
    try:
        # Verificar si se proporcionó una imagen
        if imagen is None:
            return "Por favor, sube una imagen o captura una con la cámara para realizar la predicción."
        
        estilo_predicho, probabilidad = predecir_estilo(imagen)
        
        if estilo_predicho is None:
            return "⚠️ Hubo un error al procesar la imagen. Inténtalo de nuevo."
        
        # Formatear resultado sin markdown
        resultado = f"""
        Estilo arquitectónico: {estilo_predicho}
        Probabilidad: {probabilidad:.2%}
        
        Información detallada:
        • Época histórica: {estilos_info[estilo_predicho]['epoca']}
        • Características: {estilos_info[estilo_predicho]['caracteristicas']}
        • Materiales: {estilos_info[estilo_predicho]['materiales']}
        • Relación: {estilos_info[estilo_predicho]['relacion']}
        """
        
        return resultado
    except Exception as e:
        logging.error(f"Error en predecir_estilo_gradio: {e}")
        return "⚠️ Por favor, asegúrate de subir una imagen válida. La imagen debe estar en un formato común (JPG, PNG, etc.) y no estar dañada."

def registrar_prediccion_incorrecta():
    try:
        with open('feedback_log.txt', 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Predicción Incorrecta\n")
        return "Gracias por tu retroalimentación. Hemos registrado que la predicción fue incorrecta."
    except Exception as e:
        return f"Error al registrar la retroalimentación: {str(e)}"

def registrar_prediccion_correcta():
    try:
        with open('feedback_log.txt', 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - Predicción Correcta\n")
        return "Gracias por tu retroalimentación. Hemos registrado que la predicción fue correcta."
    except Exception as e:
        return f"Error al registrar la retroalimentación: {str(e)}"

# Crear un tema personalizado oscuro
tema_personalizado = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    # Colores de fondo y texto para el modo oscuro
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    body_text_color="*neutral_200",
    body_text_color_dark="*neutral_200",
    
    # Botones
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    
    # Elementos de entrada y bloques
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_label_background_fill="*neutral_900",
    block_label_background_fill_dark="*neutral_900",
    block_label_text_color="*neutral_200",
    block_label_text_color_dark="*neutral_200",
    
    # Bordes y sombras
    block_border_width="0px",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1)",
)

with gr.Blocks(theme=tema_personalizado) as interfaz:
    gr.Markdown("<h1 style='text-align: center;'>ArchAiTect: Identificador de Estilos Arquitectónicos</h1>")
    
    gr.Markdown("""
    Para subir una imagen desde tu galería:
    1. Haz clic en 'Upload' o arrastra una imagen
    2. Haz clic en 'Enviar' para analizar la imagen
    """)
    imagen_archivo = gr.Image(
        label="Imagen", 
        type="numpy",
        height=400,
        width=600
    )
    resultado_archivo = gr.Textbox(label="Resultado", interactive=False)
    enviar_archivo = gr.Button("Enviar", size="lg")
    enviar_archivo.click(
        fn=predecir_estilo_gradio,
        inputs=imagen_archivo,
        outputs=resultado_archivo
    )

    gr.Markdown("""
    Si la predicción es incorrecta o correcta, por favor márcala usando los botones de abajo.
    """)
    
    feedback_output = gr.Textbox(label="Estado de retroalimentación", interactive=False)
    
    with gr.Row():
        btn_incorrecto = gr.Button("Predicción Incorrecta", elem_classes="flag-btn")
        btn_correcto = gr.Button("Predicción Correcta", elem_classes="flag-btn")
        
        btn_incorrecto.click(
            fn=registrar_prediccion_incorrecta,
            inputs=[],
            outputs=[feedback_output]
        )
        
        btn_correcto.click(
            fn=registrar_prediccion_correcta,
            inputs=[],
            outputs=[feedback_output]
        )

if __name__ == "__main__":
    interfaz.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 3000))
    )