# utils/helpers.py

def get_categorical_features():
 
    return [
        "Protocol", "Packet Type", "Traffic Type", "Malware Indicators",
        "Attack Type", "Attack Signature", "Action Taken", "Network Segment",
        "Alerts/Warnings"
    ]

def get_numerical_features():

    return ["Packet Length"]

def get_drop_columns():
  
    return [
        "Payload Data", "Source Port", "Destination Port",
        "IDS/IPS Alerts", "Source IP Address", "Destination IP Address",
        "User Information", "Device Information", "Geo-location Data",
        "Firewall Logs", "Proxy Information", "Log Source"
    ]

def map_day_of_week():
 
    return {
        0: "Monday", 
        1: "Tuesday", 
        2: "Wednesday", 
        3: "Thursday", 
        4: "Friday", 
        5: "Saturday", 
        6: "Sunday"
    }