import socket

class UDPSender:
    def __init__(self, ip_address="127.0.0.1", port=12345):
        self.ip_address = ip_address
        self.port = port
        self.udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    #Mandar posiçao do carro, posiçao prox checkpoint e CarAgent linha 41
    def send_data(self, car_id, position, direction):
        message = f"{car_id},{position[0]},{position[1]},{position[2]},{direction}"
        data = message.encode('utf-8')
        self.udp_client.sendto(data, (self.ip_address, self.port))

    def close(self):
        self.udp_client.close()

if __name__ == "__main__":
    pass