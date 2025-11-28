import os

# Configuración
DICCIONARIO_PATH = "diccionario.txt"
LETRAS_PROHIBIDAS = ['G', 'S', 'Z', 'Ñ', 'J', 'Q']
# Letras permitidas (según app.py): A, B, C, D, E, F, H, I, K, L, M, N, O, P, R, T, U, V, W, X, Y

# Palabras nuevas para agregar (que cumplen las reglas)
NUEVAS_PALABRAS = [
    # Nombres
    "MATEO", "DANIEL", "DAVID", "CAMILA", "VALENTINA", "PAULA", "ANDREA", "MANUEL", 
    "LEONARDO", "RAFAEL", "FABIAN", "MARIO", "PABLO", "PEDRO", "ROBERTO", "VICTOR", 
    "HECTOR", "DIANA", "CAROLINA", "CLAUDIA", "MONICA", "NATALIA", "TATIANA", "LAURA", 
    "ANA", "MARIA", "FELIPE", "RICARDO", "FERNANDO", "RAUL", "ALBERTO", "EDUARDO", 
    "ANTONIO", "MARTIN", "ADRIAN", "ALEX", "ALVARO", "ARTURO", "AURELIO", "BELEN", 
    "BERTA", "BLANCA", "BRUNO", "CANDELA", "CARMEN", "CELIA", "CLARA", "DARIO", 
    "ELENA", "ELVIRA", "EMILIO", "FABIO", "IRENE", "IVAN", "KEVIN", "LARA", "LEO", 
    "LIDIA", "LOLA", "LUCIA", "MARC", "MARINA", "MARTA", "MIRIAM", "NOA", "NOELIA", 
    "OLIVIA", "PATRICIA", "PILAR", "RAMON", "RENE", "RUBEN", "RUTH", "TAMARA", 
    "VALERIA", "VERONICA", "VICTORIA", "VIOLETA", "XAVIER",
    # Palabras comunes compatibles
    "HOLA", "MUNDO", "COMO", "BUENOS", "DIAS", "TARDE", "NOCHE", "PERDON", "AYUDA",
    "NOMBRE", "FAMILIA", "MADRE", "PADRE", "HERMANO", "HERMANA", "ABUELO", "ABUELA",
    "TIO", "TIA", "PRIMO", "PRIMA", "AMOR", "VIDA", "TIEMPO", "HORA", "HOY", "AYER",
    "MAÑANA", "TODO", "NADA", "BIEN", "MAL", "MUCHO", "POCO", "ALTO", "BAJO", "NUEVO",
    "VIEJO", "CLARO", "FRIO", "CALIENTE", "ROJO", "VERDE", "BLANCO", "NEGRO", "UNO",
    "DOS", "TRES", "CUATRO", "CINCO", "OCHO", "NUEVE", "DIEZ", "LUNES", "MARTES",
    "MIERCOLES", "JUEVES", "VIERNES", "ENERO", "FEBRERO", "ABRIL", "MAYO", "JUNIO",
    "JULIO", "NOVIEMBRE", "DICIEMBRE", "YO", "TU", "EL", "ELLA", "MI", "TU", "DE",
    "EN", "CON", "POR", "PARA", "PERO", "DONDE", "CUANDO", "PORQUE", "COMER", "BEBER",
    "DORMIR", "CORRER", "CAMINAR", "VER", "MIRAR", "OIR", "HABLAR", "LEER", "ESCRIBIR",
    "TRABAJAR", "ESTUDIAR", "APRENDER", "ENTENDER", "QUERER", "AMAR", "TENER", "HACER",
    "PODER", "IR", "VENIR", "DAR", "TOMAR", "PONER", "ABRIR", "CERRAR", "ENTRAR", "SALIR",
    "CAMA", "MESA", "SILLA", "PUERTA", "VENTANA", "TECHO", "PARED", "PISO", "BAÑO",
    "COCINA", "SALA", "COMEDOR", "ROPA", "PANTALON", "CAMISA", "ZAPATO", "DINERO",
    "PRECIO", "VALOR", "COMPRA", "VENTA", "MERCADO", "TIENDA", "BANCO", "OFICINA",
    "ESCUELA", "COLEGIO", "UNIVERSIDAD", "LIBRO", "CUADERNO", "LAPIZ", "PAPEL", "CARTA",
    "CORREO", "TELEFONO", "CELULAR", "INTERNET", "COMPUTADOR", "AUTO", "CARRO", "TREN",
    "AVION", "BARCO", "BICICLETA", "CAMINO", "CALLE", "CIUDAD", "PUEBLO", "CAMPO",
    "MONTAÑA", "RIO", "MAR", "PLAYA", "SOL", "LUNA", "CIELO", "TIERRA", "AIRE", "AGUA",
    "FUEGO", "LUZ", "COLOR", "FORMA", "TAMAÑO", "NUMERO", "LETRA", "PALABRA", "IDEA",
    "VERDAD", "MENTIRA", "MIEDO", "DOLOR", "PLACER", "ALEGRIA", "TRISTEZA", "ENOJO",
    "CALMA", "PAZ", "GUERRA", "LIBERTAD", "LEY", "ORDEN", "PODER", "FUERZA", "ENERGIA",
    "SALUD", "ENFERMEDAD", "MEDICO", "DOCTOR", "HOSPITAL", "REMEDIO", "CURA", "CUERPO",
    "CABEZA", "MANO", "PIE", "BRAZO", "PIERNA", "OJOS", "BOCA", "NARIZ", "OREJA", "PELO",
    "CARA", "DIENTE", "LENGUA", "DEDO", "UÑA", "PIEL", "HUESO", "SANGRE", "CORAZON",
    "MENTE", "ALMA", "ESPIRITU", "DIOS", "FE", "ARTE", "MUSICA", "PINTURA", "CINE",
    "TEATRO", "BAILE", "FIESTA", "JUEGO", "DEPORTE", "FUTBOL", "PELOTA", "EQUIPO",
    "GRUPO", "LIDER", "JEFE", "EMPLEADO", "OBRERO", "ALUMNO", "MAESTRO", "PROFESOR",
    "POLICIA", "LADRON", "JUEZ", "ANIMAL", "PERRO", "GATO", "CABALLO", "VACA", "POLLO",
    "PAJARO", "PEZ", "LEON", "TIGRE", "OSO", "LOBO", "ARBOL", "FLOR", "FRUTA", "MANZANA",
    "PERA", "UVA", "LIMON", "NARANJA", "BANANO", "TOMATE", "PAPA", "ARROZ", "PAN",
    "LECHE", "CARNE", "HUEVO", "CAFE", "VINO", "AGUA", "AIRE", "TIERRA", "FUEGO"
]

def filtrar_palabra(palabra):
    palabra = palabra.upper().strip()
    if not palabra:
        return False
    for letra in LETRAS_PROHIBIDAS:
        if letra in palabra:
            return False
    return True

def main():
    palabras_finales = set()

    # 1. Leer diccionario existente
    if os.path.exists(DICCIONARIO_PATH):
        with open(DICCIONARIO_PATH, 'r', encoding='utf-8') as f:
            for linea in f:
                palabra = linea.strip().upper()
                if filtrar_palabra(palabra):
                    palabras_finales.add(palabra)
    
    # 2. Agregar nuevas palabras
    for palabra in NUEVAS_PALABRAS:
        palabra = palabra.upper()
        if filtrar_palabra(palabra):
            palabras_finales.add(palabra)

    # 3. Guardar ordenado
    lista_ordenada = sorted(list(palabras_finales))
    
    with open(DICCIONARIO_PATH, 'w', encoding='utf-8') as f:
        for palabra in lista_ordenada:
            f.write(palabra + '\n')
            
    print(f"✅ Diccionario actualizado. Total palabras: {len(lista_ordenada)}")
    print(f"🚫 Letras eliminadas: {LETRAS_PROHIBIDAS}")

if __name__ == "__main__":
    main()
