import paho.mqtt.client as mqtt


class MQTTClient:
    """
    Classe para criar um cliente MQTT.
    """
    broker = "mqtt.ect.ufrn.br"
    port = 1883
    username = "mqtt"
    password = "lar_mqtt"

    def __init__(self):
        """
        Inicializa o cliente MQTT com as credenciais fornecidas.
        """
        self.client = mqtt.Client()

    def on_connect(self, client, userdata, flags, rc):
        """
        Callback para quando o cliente se conecta ao broker.
        """
        if rc == 0:
            print("Conectado ao broker MQTT")
        else:
            print("Falha na conexão. Código de retorno:", rc)

    def on_disconnect(self, client, userdata, rc):
        """
        Callback para quando o cliente se desconecta do broker.
        """
        print("Desconectado do broker MQTT")

    def connect(self):
        """
        Conecta o cliente ao broker MQTT.
        """
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.connect(self.broker, self.port)
        self.client.loop_start()

    def publish(self, topic, message):
        """
        Publica uma mensagem em um tópico específico.
        """
        result, _ = self.client.publish(topic, message)
        if result == mqtt.MQTT_ERR_SUCCESS:
            print("Mensagem enviada com sucesso")
        else:
            print("Falha ao enviar mensagem. Código de retorno:", result)

    def disconnect(self):
        """
        Desconecta o cliente do broker MQTT.
        """
        self.client.loop_stop()
        self.client.disconnect()

    @classmethod
    def create_and_publish(cls, topic, message):
        """
        Cria uma instância, conecta e publica uma mensagem.
        """
        instance = cls()
        instance.connect()
        instance.publish(topic, message)
        instance.disconnect()
        return instance
