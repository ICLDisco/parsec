## Setting up a Jenkins server for testing PaRSEC

1. Set up the Jenkins server as suggested on the Jenkins webpage

2. Create a PaRSEC configuration file on the Jenkins server. This file should be located in $JENKINS_HOME and named parsec.jenkins.conf. At the minimum it contains a
<pre>
userName = The bitbucket login to access the PaRSEC repo
userPassword = The corresponding password.
useHipChat = true
</pre>

3. To run the PaRSEC groovy script the following set of permission must be added to Jenkins scriptApproval.xml.
<pre>
<approvedSignatures>
  <string>method groovy.lang.GroovyObject invokeMethod java.lang.String java.lang.Object</string>
  <string>method java.lang.AutoCloseable close</string>
  <string>method java.lang.String getBytes</string>
  <string>method java.net.HttpURLConnection disconnect</string>
  <string>method java.net.HttpURLConnection getResponseCode</string>
  <string>method java.net.HttpURLConnection setRequestMethod java.lang.String</string>
  <string>method java.net.URL openConnection</string>
  <string>method java.net.URLConnection connect</string>
  <string>method java.net.URLConnection getInputStream</string>
  <string>method java.net.URLConnection setDoOutput boolean</string>
  <string>method java.net.URLConnection setRequestProperty java.lang.String java.lang.String</string>
  <string>method java.util.Dictionary get java.lang.Object</string>
  <string>method java.util.Dictionary put java.lang.Object java.lang.Object</string>
  <string>method java.util.Properties load java.io.InputStream</string>
  <string>method java.util.regex.Matcher find</string>
  <string>method java.net.HttpURLConnection getErrorStream</string>
  <string>new java.io.File java.lang.String</string>
  <string>new java.util.Properties</string>
  <string>new java.io.InputStreamReader java.io.InputStream</string>
  <string>new java.io.InputStreamReader java.io.InputStream java.lang.String</string>
  <string>staticMethod java.lang.System getenv java.lang.String</string>
  <string>staticMethod org.codehaus.groovy.runtime.DefaultGroovyMethods getProperties java.lang.Object</string>
  <string>staticMethod org.codehaus.groovy.runtime.DefaultGroovyMethods getText java.io.InputStream</string>
  <string>staticMethod org.codehaus.groovy.runtime.DefaultGroovyMethods hasProperty java.lang.Object java.lang.String</string>
  <string>staticMethod org.codehaus.groovy.runtime.DefaultGroovyMethods newDataInputStream java.io.File</string>
  <string>staticMethod org.codehaus.groovy.runtime.EncodingGroovyMethods encodeBase64 byte[]</string>
</approvedSignatures>
</pre>

4. For a streamlined integration with HipChat you might want to install the [Jenkins HipChat plugin](https://plugins.jenkins.io/hipchat)
