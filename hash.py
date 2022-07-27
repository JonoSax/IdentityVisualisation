import hashlib

m = hashlib.sha256()

identityFile = open("identityInfo.csv", "r").readlines()
entitlementFile = open("entitlementInfo.csv", "r").readlines()

identityHash = open("identityHash.csv", "w")
entitlementHash = open("entitlementHash.csv", "w")

def hashit(text):

    return hashlib.sha256(text[:-1].encode()).hexdigest()

for i in identityFile:
    if i == "": hashed = ""
    else: hashed = hashit(i)
    identityHash.write(hashit(i) + "\n")
identityHash.close()

for h in entitlementFile:
    if h == "": hashed = ""
    else: hashed = hashit(h)
    entitlementHash.write(hashed + "\n")
entitlementHash.close()

#d54f5c92748c66a177f4aef2f68986f6668692fe4cc92598e063a134c3475d08