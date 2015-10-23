#! /bin/bash
#
# This script provides an easy way to execute an arbitrary java class
# in your maven classpath.  You may do similar with mvn exec:java, but
# consider:
#
# exec:java does not fork a new jvm, so it may do strange things, like
# refuse to die if there are daemon threads, or use extra heap memory
# for the rest of its processing.
#
# exec:java requires you to write command line args with
# -Dexec.args="arg1 arg2 ...", and that means you cannot use command
# line tab completion.
#
# exec:java is tied up with default maven logging, which is written to
# standard output.  If your program writes to standard output you'll
# have to filter out maven's log statements.
#
# You could write an appassembler configuration, but you'd need one
# for every main class you want to run, which is not friendly for
# adhoc command lines.  Appassembler will also copy your jars - you
# can't use them in place.
#
# To generate the classpath.txt file, add this to your pom.xml:
#
# <plugin>
#   <groupId>org.apache.maven.plugins</groupId>
#   <artifactId>maven-dependency-plugin</artifactId>
#   <version>2.8</version>
#   <executions>
#     <execution>
#       <id>build-classpath</id>
#       <phase>generate-sources</phase>
#       <goals>
#         <goal>build-classpath</goal>
#       </goals>
#       <configuration>
#         <outputFile>${project.build.directory}/classpath.txt</outputFile>
#       </configuration>
#     </execution>
#   </executions>
# </plugin>
#
# If you don't want to configure that in your pom.xml, you can create
# the classpath.txt file from the command line like this:
#
# $ mvn dependency:build-classpath -Dmdep.outputFile=target/classpath.txt

core_dir=$(dirname $0)/../core
CP=${core_dir}/target/classes:$(cat ${core_dir}/target/classpath.txt)

if [ $# -lt 1 -o ! -f ${core_dir}/target/classpath.txt ]
then
    echo "Usage: $0 fully.qualified.Classname"
    echo "$core_dir/target/classpath.txt must exist"
    exit 1
fi

# The -Dfile.encoding=UTF-8 is for Mac, where the default encoding is
# MacRoman *even* if your locale is UTF-8.  The Mac jvm folks, for
# compatibility, think that they should not listen to the system
# locale for the encoding.  At least in Java 1.6.  Another way around
# this is to set JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF-8 in the
# environment, but then all your java tools will echo an annoying line
# saying "Picked up JAVA_TOOL_OPTIONS: .."
java ${JVM_ARGS} -Dfile.encoding=UTF-8 -cp "${CP}" "$@"
