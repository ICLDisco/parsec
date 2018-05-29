import groovy.json.JsonSlurper
import hudson.util.RemotingDiagnostics
import jenkins.model.Jenkins
import java.util.regex.Pattern
import java.util.regex.Matcher

@groovy.transform.Field Properties config

// https://qa.nuxeo.org/jenkins/pipeline-syntax/globals

def PROJECT_NAME = "project_name"

//node {
//    sh 'env > env.txt' 
//    for (String i : readFile('env.txt').split("\r?\n")) {
//        println i
//    }
//}

config = new Properties()
//String configFileName = System.getenv('JENKINS_HOME') + "/parsec.jenkins.conf"
// For some obscure reason the JENKINS_HOME is always NULL
String configFileName = "${env.JENKINS_HOME}/parsec.jenkins.conf"
try {
    File configFile = new File(configFileName)
    config.load(configFile.newDataInputStream())
} catch (Exception ex) {
    println "Exception: " + ex.getMessage()
    error("Cant load local configuration file ${configFileName} (required for connection to bitbucket)")
}

propertiesData = [disableConcurrentBuilds()]
properties(propertiesData)

node {
    def regex = '^https://bitbucket\\.org/(.*?)/(.*?)/pull-requests/(\\d+)'
    try {
        def l = Pattern.compile(regex).matcher(env.CHANGE_URL)
        println "${l.group(1)} -> ${l.group(2)}"
        config.put("organization", l.group(1))
        config.put("repository", l.group(2))
    } catch (Exception ex) {
        println "Exception: " + ex.getMessage() + " (${env.CHANGE_URL})"
        println "Cant identify the organization and repository from ${env.CHANGE_URL} using ${regex}"
        config.put("organization", "icldistcomp")
        config.put("repository", "parsec")
    }
    if( !config.hasProperty('useHipChat') ) {
        config.put("useHipChat", false)
    }
}

// Save the master of the lack check
//git log --format="%H" -n 1 master

//currentBuild.properties.each { println "currentBuild.${it.key} -> ${it.value}" }
//propertiesData.properties.each { println "$it.key -> $it.value" }
//config.each { println "config.${it.key} -> ${it.value}" }

pipeline {
    agent any
    stages {
        stage ('Clone') {
            when {
                beforeAgent true  // execute the when clause early
                anyOf {
                    expression {
                        // https://github.com/jenkinsci/jenkins/blob/master/core/src/main/java/hudson/model/Result.java
                        return isApprovedBranch(config.repository, env.CHANGE_ID)
                    }
                }
            }
            steps {
                script {
                    try {
                        hipchatSend color: 'YELLOW', notify: false, room: "CI", sendAs: "Sauron",
                            message: "Starting: <a href=\"${env.BUILD_URL}\">${env.JOB_NAME} [${env.BUILD_NUMBER}]</a><br>" +
                                     "Pull Request <a href=\"${env.CHANGE_URL}\">#${env.CHANGE_ID}</a>."
                    } catch (Exception ex) {  //disable HipChat
                        config.useHipChat = false
                    }
                }
                checkout scm
            }
        }
        stage ('Build and Test') {
            environment {
                BUILDDIR = "build"
                DPLASMA_PRECISIONS="s\\;d\\;c\\;z"
            }
            parallel {
               stage ('Debug all') {
                    environment {
                        BUILDTYPE = "Debug"
                        BUILDDIR = "${env.BUILDDIR}/${env.BUILDTYPE}"
                    }
                    options {
                        timeout(time:1, unit: 'HOURS')
                    }
                    steps {
                        sh "echo 'Prepare for a full Debug build'"
                        sh "mkdir -p ${env.BUILDDIR}"
                        dir (env.BUILDDIR) {
                            sh "${env.WORKSPACE}/contrib/jenkins/script.saturn ${env.BUILDTYPE}"
                        }
                    }
                }
                stage ('Release all') {
                    environment {
                        BUILDTYPE = "Release"
                        BUILDDIR = "${env.BUILDDIR}/${env.BUILDTYPE}"
                    }
                    options {
                        timeout(time:1, unit: 'HOURS')
                    }
                    steps {
                        sh "echo 'Prepare for a full Debug build'"
                        sh "mkdir -p ${env.BUILDDIR}"
                        dir (env.BUILDDIR) {
                            sh "${env.WORKSPACE}/contrib/jenkins/script.saturn ${env.BUILDTYPE}"
                        }
                    }
                }
            }
        }
    }
    post { 
        // no always
        regression {
            hipchatSend color: 'RED', notify: true, room: "CI", sendAs: "Sauron",
                message: "REGRESSION: Job <a href=\"${env.BUILD_URL}\">${env.JOB_NAME} [${env.BUILD_NUMBER}]</a><br>" +
                         "Pull Request <a href=\"${env.CHANGE_URL}\">#${env.CHANGE_ID}</a>."
            //emailext (
            //    subject: "STARTED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'",
            //    body: """<p>STARTED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]':</p>
            //  <p>Check console output at &QUOT;<a href='${env.BUILD_URL}'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",
            //    recipientProviders: [[$class: 'DevelopersRecipientProvider']]
            //)
        }
        success {
            hipchatSend color: 'GREEN', notify: true, room: "CI", sendAs: "Sauron",
                message: "SUCCESS: Job <a href=\"${env.BUILD_URL}\">${env.JOB_NAME} [${env.BUILD_NUMBER}]</a><br>" +
                         "Pull Request <a href=\"${env.CHANGE_URL}\">#${env.CHANGE_ID}</a> ready to merge"
        }
        failure {
            hipchatSend color: 'RED', notify: true, room: "CI", sendAs: "Sauron",
                message: "FAILURE: <a href=\"${env.BUILD_URL}\">${env.JOB_NAME} [${env.BUILD_NUMBER}]</a><br>" +
                         "Pull Request <a href=\"${env.CHANGE_URL}\">#${env.CHANGE_ID}</a> consistently fails.<br>" +
                         "This kind of consistency is N.O.T. good"
        }
    }
}

// Find the PaRSEC version from the CMakeLists.txt file
String getVersion() {
    try {
        def p = ['grep', '(PARSEC_VERSION_', 'CMakeLists.txt'].execute() | ['awk', '/^set/ {printf \"%d.\", $3}'].execute()
        p.waitFor()
        if( 0 == p.exitValue() ) {
            return p.in.text[0..-2]
        } else {
            println "ERROR: ${p.exitValue()} -> ${p.err.text}"
        }
    } catch (err) {
        println "ERROR: while retrieving PaRSEC version from CMakeLists.txt"
        throw err
    }
    return "0.0"
}

def displayServerResponse( InputStream is ) {
    BufferedReader input = new BufferedReader(new InputStreamReader(is))
    String inputLine;
    while ((inputLine = input.readLine()) != null) 
        println inputLine
    input.close();
}

// Check if the branch author has validated his work by approving
// the branch.
def isApprovedBranch(repository, pr) {
    Boolean prApproved = false
    String userpass = config.userName + ":" + config.userPassword;
    String basicAuth = "Basic " + userpass.bytes.encodeBase64().toString()

    def url = "https://api.bitbucket.com/2.0/repositories/${config.organization}/${repository}/pullrequests/${pr}/"

    def conn = new URL(url).openConnection()
    conn.setRequestProperty('Authorization', basicAuth)

    try {
        statusCode = conn.getResponseCode()
        println "Connection status code: $statusCode "
        if (statusCode==401) {
            println "Not authorized"
            response=displayServerResponse(connection.getErrorStream())
        }
        if (statusCode==200) {
            InputStream inputStream = conn.getInputStream()
            def names = new groovy.json.JsonSlurper().parseText(inputStream.text)
            //names.each{ k, v ->
            //    println "element name: ${k}, element value: ${v}"
            //}
            inputStream.close()
            names['participants'].each {
                if( it['approved'] ) {
                    if( it['user']['username'] == names['author']['username'] ) {
                        prApproved = true
                        println "Pullrequest ${pr} approved by its author ${it['user']['username']}"
                    }
                }
            }
        }
        if (statusCode==400) {
            println "Bad request"
            println "Server response:"
            println "-----"
            response=displayServerResponse(connection.getErrorStream())
            println "-----"
        }
    } catch (Exception e) {
        println "Error connecting to the URL"
        println e.getMessage()
    } finally {
        if (conn != null) {
            conn.disconnect();
        }
    }
    return prApproved
}

