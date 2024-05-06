from machine import Pin, Timer
import network
import socket
import time

class ESP32S2Mini:
    def __init__(self, ssid, password, udp_port=8080):
        self.ssid = ssid
        self.password = password
        self.udp_port = udp_port
        
        self.led_pin = 2  # assuming LED connected to GPIO pin 2
        self.forward_pin = 33
        self.backward_pin = 35
        self.left_pin = 37
        self.right_pin = 39

        self.movement_delay_ms = 100
        self.delay_change = 10

        self.led = Pin(self.led_pin, Pin.OUT)
        self.forward_pin = Pin(self.forward_pin, Pin.OUT)
        self.backward_pin = Pin(self.backward_pin, Pin.OUT)
        self.left_pin = Pin(self.left_pin, Pin.OUT)
        self.right_pin = Pin(self.right_pin, Pin.OUT)

        self.led.value(0)  # Turn off LED initially

        # Connect to WiFi
        self.wifi = network.WLAN(network.STA_IF)
        self.wifi.active(True)
        self.wifi.connect(self.ssid, self.password)
        while not self.wifi.isconnected():
            pass
        print("Connected to WiFi")

        # Start UDP server
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind(('0.0.0.0', self.udp_port))
        print("UDP server started")

        self.flash_timer = Timer(0)
        self.flash_timer.init(period=0, mode=Timer.PERIODIC, callback=self._flash_led)

    def _flash_led(self, timer):
        for _ in range(3):
            self.led.value(1)
            time.sleep_ms(200)
            self.led.value(0)
            time.sleep_ms(200)
        self.led.value(0)

    def move_car(self, target_pin):
        target_pin.on()
        time.sleep_ms(self.movement_delay_ms)
        target_pin.off()

    def handle_udp_messages(self):
        while True:
            data, addr = self.udp.recvfrom(1024)
            print("Received message:", data.decode())

            if b"mf" in data:
                print("Moving forward!")
                self.move_car(self.forward_pin)
            elif b"mb" in data:
                print("Moving backward!")
                self.move_car(self.backward_pin)
            elif b"ml" in data:
                print("Moving left!")
                self.left_pin.on()
                self.right_pin.off()
            elif b"mr" in data:
                print("Moving right!")
                self.right_pin.on()
                self.left_pin.off()
            elif b"rr" in data:
                print("Left/Right Reset!")
                self.right_pin.off()
                self.left_pin.off()
            elif b"+d" in data:
                self.movement_delay_ms += self.delay_change
                print("Delay increased to", self.movement_delay_ms)
            elif b"-d" in data:
                self.movement_delay_ms -= self.delay_change
                print("Delay decreased to", self.movement_delay_ms)
