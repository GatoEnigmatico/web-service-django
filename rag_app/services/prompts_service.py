import json

# Definimos constantes para los keys principales
KEY_PERSONALIDAD = "personalidad"
KEY_PROPOSITOS = "propositos"
KEY_DINAMICA = "dinamica"
KEY_INTERACCION = "interaccion"

# Definimos constantes para los propósitos
PROP_INICIALIZAR_CONVERSACION = "inicializar_conversacion"
PROP_RESPONDER_AL_USUARIO = "responder_al_usuario"
PROP_REALIZAR_PREGUNTA = "realizar_pregunta"

# Diccionario con los propósitos y sus descripciones
PROPOSITOS_DICT = {
    PROP_INICIALIZAR_CONVERSACION: "Inicias la conversación de manera cálida y accesible, estableciendo confianza con el usuario.",
    PROP_RESPONDER_AL_USUARIO: "Respondes de manera natural y auténtica, asegurándote de reconocer el mensaje del usuario con empatía.",
    PROP_REALIZAR_PREGUNTA: "Formulas preguntas de manera fluida y relevante para que la conversación siga desarrollándose de manera orgánica.",
}

def build_system_prompt():
    """
    Construye un prompt con una estructura definida y agrega propósitos según las claves proporcionadas.
    """
    prompt_json = {
    KEY_PERSONALIDAD: """Eres una persona carismática y con gran habilidad para conectar con los demás.
    Inicias la conversación con un tono amigable y relajado, haciendo que el usuario se sienta escuchado y cómodo desde el primer momento.
    Aprovecha el contexto para hacer la charla interesante, con un toque de curiosidad y empatía.
    Mantén la interacción dinámica, con respuestas que inviten a seguir conversando sin sentirse mecánicas o estructuradas."""
    }


    # Reconstrucción del prompt en formato de texto
    final_prompt = (
        f"{prompt_json[KEY_PERSONALIDAD]}\n\n"
        # + "\n".join(prompt_json[KEY_PROPOSITOS])
        # + "\n\n"
        # f"{prompt_json[KEY_DINAMICA]} {prompt_json[KEY_INTERACCION]}"
    )

    return final_prompt

def build_prompt(user_prompt,  question):
    """
    Construye un prompt con una estructura definida y agrega propósitos según las claves proporcionadas.
    """
    prompt_json = {
        KEY_PROPOSITOS: [
            f'Responde de manera auténtica a lo que dice el usuario: "{user_prompt}", asegurándote de reconocer su mensaje de forma natural.',
            f'Luego, guía la conversación sin que se sienta forzada, integrando la pregunta: "{question}" de manera sutil y fluida.',
        ],
        # KEY_DINAMICA: "Aprovecha el contexto para hacer la charla interesante, con un toque de curiosidad y empatía.",
        # KEY_INTERACCION: "Mantén la interacción dinámica, con respuestas que inviten a seguir conversando sin sentirse mecánicas o estructuradas.",
    }

    # Agregar propósitos adicionales según las claves proporcionadas
    # prompt_json = add_propositos(prompt_json, propositos_keys)

    # Reconstrucción del prompt en formato de texto
    final_prompt = (
        # f"{prompt_json[KEY_PERSONALIDAD]}\n\n"
        ""
        + "\n".join(prompt_json[KEY_PROPOSITOS])
        # + "\n\n"
        # f"{prompt_json[KEY_DINAMICA]} {prompt_json[KEY_INTERACCION]}"
    )
    return "\n".join(prompt_json[KEY_PROPOSITOS])
    return final_prompt


def add_propositos(existing_prompt, propositos_keys):
    """
    Agrega propósitos adicionales al array 'propositos' en base a las claves proporcionadas.

    :param existing_prompt: JSON con la estructura base del prompt.
    :param propositos_keys: Lista de claves de los propósitos a agregar.
    :return: Prompt actualizado con los nuevos propósitos.
    """
    nuevos_propositos = [
        PROPOSITOS_DICT[key] for key in propositos_keys if key in PROPOSITOS_DICT
    ]

    # Añadir los nuevos propósitos al array existente
    existing_prompt[KEY_PROPOSITOS].extend(nuevos_propositos)

    return existing_prompt
