
'''
What it is: UDP Client

What it does: 
    1) Creates a Socket with certain parameters
    2) Reads XML File
    3) Encodes XML File
    4) Sends the Encoded XML Data to the Server (v2.0)
    3) Waits for the Server's Reply
'''

#%%
import socket
import os
import traceback
import datetime
#-----------------------------------------------------------------------------
#Connection Parameters:
port = 54321
port = 8080


#R = input(">>Ler IP de Ficheiro? (1-yes)(0-no) ")
#R = int(R)
error = True
while error:
    try:
        fid = open("ip.txt",'r',encoding = "utf-8")
        enderecoIP = fid.read()
        enderecoIP = enderecoIP.replace("\n",'')
        error = False
        break
    except FileNotFoundError:
        print("File 'ip.txt' does not exit")
        print("creating one ...")
        with open("ip.txt","w+") as fid:
            pass
        #end-with
    #end-try-except
    
    R = input(">>IP LocalHost? (1-yes)(0-no) ")
    R = int(R)
    
    if (R==1):
        enderecoIP = "192.168.137.172"
    else:
        enderecoIP = input(">> IP: ")
    #end-if-else
    
    with open("ip.txt","w+") as fid:
        fid.write(enderecoIP)
    #end-with
#end-while

print("IP: ", enderecoIP)
print("Port: ", port)
serverAddressPort = (enderecoIP, port)
bufferSize = 1024
#-----------------------------------------------------------------------------


# Create a UDP socket at client sidesample3
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)


#-----------------------------------------------------------------------------
def sendMessage(XMLdata):

    bytesToSend = str.encode(XMLdata) #Encode Message to Bytes
    
    # Send to server using created UDP socket
    print("Sending message to Server...")
    UDPClientSocket.sendto(bytesToSend, serverAddressPort)
    print("Message sent.\n")
    return

#-----------------------------------------------------------------------------
# Waits for Server's REPLY:
def receiveReply():
    print("Waiting for Server's REPLY...")
    msgFromServer = UDPClientSocket.recvfrom(bufferSize)
    
    
    #Display Received Message:
    msg = "Message from Server {}".format(msgFromServer[0])
    print(msg)
    return
#-----------------------------------------------------------------------------


def funclear():
    os.system('cls' if os.name == 'nt' else 'clear')
    return

def showIP(enderecoIP):
    print("\nIP: ", enderecoIP)
    return

def showPort(port):
    print("\nPort: ", port)
    return



#If this flag is true:
#Move Left becomes Move Right
#and Move Right becomes Move Left
reverseCarDirectionCommand = True

while True:
    try:
        msgContent = input("> Message > ")
        if (reverseCarDirectionCommand):
            if (msgContent == "mr"):
                msgContent = "ml"
            elif (msgContent == "ml"):
                msgContent = "mr"
            else:
                pass
            #end-if-else
        #end-if-else
        
        if (msgContent.lower() == "clear"):
            funclear()
        elif (msgContent.lower() == "showip"):
            showIP()
        elif (msgContent.lower() == "showport"):
            showPort()
        elif (msgContent.lower() == "stop" or msgContent.lower() == "exit"):
            break

        else:
            pass
        #end-if-else
            
            
        sendMessage(msgContent)
    except:
        print(traceback.format_exc())