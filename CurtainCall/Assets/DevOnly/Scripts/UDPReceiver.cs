using System;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;

public class UDPReceiver : MonoBehaviour
{
    private UdpClient udpClient;
    private IPEndPoint ipEndPoint;
    public List<(float, float, float)> handLandmarkList = new List<(float, float, float)>(21);

    private async void Start()
    {
        udpClient = new UdpClient(5005);
        ipEndPoint = new IPEndPoint(IPAddress.Any, 0);

        while (true)
        {
            try
            {
                // receive position data via UDP
                UdpReceiveResult result = await udpClient.ReceiveAsync();
                byte[] data = result.Buffer;
                
                // float x = System.BitConverter.ToSingle(data, 0);
                // float y = System.BitConverter.ToSingle(data, 4);
                // float z = System.BitConverter.ToSingle(data, 8);
                //
                // Debug.Log($"landmark : {x}, {y}. {z}");
                
                for (int i = 0; i < 21; i++)
                {
                    float x = System.BitConverter.ToSingle(data, 0 + 12 * i);
                    float y = System.BitConverter.ToSingle(data, 4 + 12 * i);
                    float z = System.BitConverter.ToSingle(data, 8 + 12 * i);
                
                    handLandmarkList[i] = (x, y, z);
                    Debug.Log($"landmark : {x}, {y}. {z}");
                }
            }
            catch (SocketException e)
            {
                Debug.LogError(e);
            }
        }
    }

    private void OnApplicationQuit()
    {
        udpClient.Close();
    }
}