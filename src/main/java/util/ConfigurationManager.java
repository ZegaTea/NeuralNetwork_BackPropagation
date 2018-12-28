package util;

import org.apache.commons.configuration.ConfigurationException;
import org.apache.commons.configuration.PropertiesConfiguration;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import java.io.File;


public class ConfigurationManager {
	private static final String CONFIG_FILE = "config.properties";
	private PropertiesConfiguration config;
	private static ConfigurationManager conf = new ConfigurationManager();

	static Logger logger = Logger.getLogger(ConfigurationManager.class);
	
	private ConfigurationManager() {
		config = new PropertiesConfiguration();
		loadconfig();
		loadlog();
	}

	private void loadconfig() {
		File co = new File(CONFIG_FILE);
		try {
			config.setFile(co);
			config.load();
		} catch (ConfigurationException e) {
			logger.error("",e);
		}
	}

	public void loadlog() {
		PropertyConfigurator.configure("log4j.properties");
	}

	public static ConfigurationManager getInstance() {
		return conf;
	}

	public String getString(String name) {
		return config.getString(name);
	}
	

}
