<?xml version="1.0"?>
<project name="Common Build Properties">

	<!-- Common properties that allow overrides -->

	<property name="rootbuild.dir" 		value="${directory::get-current-directory()}" />
		
	<property name="build.dest.dir" 	value="${rootbuild.dir}/build/${framework::get-target-framework()}" />
	<property name="build.working.dir" 	value="${rootbuild.dir}/bin" />

	<property name="strong_name"		value="NANT_STRONGNAME"	unless="${property::exists('strong_name')}" />
	<property name="debug"				value="false"			unless="${property::exists('debug')}" />
	<property name="optimize"			value="true"			unless="${property::exists('optimize')}" />
	<property name="doc.dir"			value="${base.dir}/docs" />
	<property name="nunit.formatter"	value="Plain"			unless="${property::exists('nunit.formatter')}" />
	<property name="os" 				value="${operating-system::to-string(environment::get-operating-system())}" />
	<property name="isWindows" 			value="${string::starts-with(os, 'Microsoft Windows')}" />
	<property name="isMono" 			value="${string::starts-with(framework::get-target-framework(),'mono')}" />
	<property name="target.clr" 		value="${framework::get-target-framework()}" />
	
	<if test="${framework::get-target-framework()=='netcf-1.0'}">
	    <fail message=".NET-CF is not supported. Please use the .NET Framework or Mono." />
	</if>
    
	<if test="${framework::get-target-framework()=='net-1.1' or framework::get-target-framework()=='mono-1.0'}">
	    <property name="dotnet_define"  value="DOTNET1" />
	</if>
	<if test="${not (framework::get-target-framework()=='net-1.1') and not (framework::get-target-framework()=='mono-1.0')}">
	    <property name="dotnet_define"  value="DOTNET2" />
	</if>


	<!-- find out where nunit.framework.dll is -->

	<property name="lib.dir"
		value="${path::combine(nant::get-base-directory(), 'lib')}"
		dynamic="true" />
		
	<property name="lib.family.dir"
		value="${path::combine(lib.dir,framework::get-family(framework::get-target-framework()))}"
		dynamic="true" />
	
	<!-- for nant 0.85 rc2 or higher -->
	
	<property name="lib.framework.dir"
		value="${path::combine(lib.family.dir, version::to-string(framework::get-version(framework::get-target-framework())))}"
		dynamic="true" />

    <!-- Uncomment this block for nant backward compatibility
	<property name="lib.framework.dir"
		value="${path::combine(lib.family.dir, framework::get-version(framework::get-target-framework()))}"
		dynamic="true" />
    -->
		
	<property name="nunit.framework.dll"
			value="${path::combine(lib.framework.dir, 'nunit.framework.dll')}" />

	<property name="nant.tasks.nunit2"
			value="False"
			unless="${property::exists('nant.tasks.nunit2')}" />

</project>
