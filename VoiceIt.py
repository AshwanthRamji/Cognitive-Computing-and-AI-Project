import hashlib,requests

class VoiceIt:
    developerId = ""
    urlUsers = 'https://siv.voiceprintportal.com/sivservice/api/users'
    urlEnrollments = 'https://siv.voiceprintportal.com/sivservice/api/enrollments'
    urlAuthentication = 'https://siv.voiceprintportal.com/sivservice/api/authentications'

    def getSHA256(self, strData):
        return hashlib.sha256(strData.encode("ascii")).hexdigest()

    def __init__(self, devId):
        self.developerId = devId

    def createUser(self, userId, passwd):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'application/json', "UserId": userId, "VsitPassword": password, "VsitDeveloperId": self.developerId}
        try:
            response = requests.post(self.urlUsers, headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def deleteUser(self, userId, passwd):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'application/json', "UserId": userId,
                   "VsitPassword": password, "VsitDeveloperId": self.developerId}
        try:
            response = requests.delete(self.urlUsers, headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def getUser(self, userId, passwd):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'application/json', "UserId": userId,
                   "VsitPassword": password, "VsitDeveloperId": self.developerId}
        try:
            response = requests.get(self.urlUsers, headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def createEnrollment(self, userId, passwd, pathToEnrollmentWav, contentLanguage=""):
        password = self.getSHA256(passwd)
        
        with open(pathToEnrollmentWav, 'rb') as file:
            wavData = file.read()
        
        headers = {'PlatformID': '2', 'Content-Type': 'audio/wav', "UserId": userId,
                   "VsitPassword": password, "VsitDeveloperId": self.developerId, "ContentLanguage":contentLanguage}
        try:
            response = requests.post(
                self.urlEnrollments, headers=headers, data=wavData)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def createEnrollmentByWavURL(self, userId, passwd, urlToEnrollmentWav, contentLanguage=""):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'audio/wav', "UserId": userId, "VsitPassword": password,
                   "VsitDeveloperId": self.developerId, "VsitwavURL": urlToEnrollmentWav, "ContentLanguage":contentLanguage}
        try:
            response = requests.post(
                self.urlEnrollments + "/bywavurl", headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def getEnrollments(self, userId, passwd):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'application/json', "UserId": userId,
                   "VsitPassword": password, "VsitDeveloperId": self.developerId}
        try:
            response = requests.get(self.urlEnrollments, headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def deleteEnrollment(self, userId, passwd, enrollmentId):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'application/json', "UserId": userId,
                   "VsitPassword": password, "VsitDeveloperId": self.developerId}
        try:
            response = requests.delete(
                self.urlEnrollments + "/" + enrollmentId, headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def authentication(self, userId, passwd, pathToAuthenticationWav, contentLanguage=""):
        password = self.getSHA256(passwd)
        
        with open(pathToAuthenticationWav, 'rb') as file:
            wavData = file.read()
        
        headers = {'PlatformID': '2', 'Content-Type': 'audio/wav', "UserId": userId, "VsitPassword": password, "VsitDeveloperId": self.developerId, "ContentLanguage":contentLanguage}
        try:
            response = requests.post(
                self.urlAuthentication, headers=headers, data=wavData)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()

    def authenticationByWavURL(self, userId, passwd, urlToAuthenticationWav, contentLanguage=""):
        password = self.getSHA256(passwd)
        headers = {'PlatformID': '2', 'Content-Type': 'audio/wav', "UserId": userId, "VsitPassword": password, "VsitDeveloperId": self.developerId, "VsitwavURL": urlToAuthenticationWav, "ContentLanguage":contentLanguage}
        try:
            response = requests.post(
                self.urlAuthentication + "/bywavurl", headers=headers)
            return response.text
        except requests.exceptions.HTTPError as e:
            return e.read()
