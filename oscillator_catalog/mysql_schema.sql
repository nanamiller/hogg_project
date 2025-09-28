-- DROP USER 'nana'@'localhost';
-- FLUSH PRIVILEGES;
-- CREATE USER 'nana'@'%';

-- CREATE DATABASE IF NOT EXISTS stars_db;
-- USE stars_db;
-- GRANT ALL ON stars_db.* TO 'nana'@'%';


DROP TABLE IF EXISTS dataset;
CREATE TABLE IF NOT EXISTS dataset (
    dataset_id VARCHAR(32),
    description TEXT,
    PRIMARY KEY (dataset_id)
);

DROP TABLE IF EXISTS star;
CREATE TABLE IF NOT EXISTS star (
    star_id VARCHAR(32),
    PRIMARY KEY (star_id)
);

DROP TABLE IF EXISTS task;
CREATE TABLE IF NOT EXISTS task (
    star_id VARCHAR(32),
    dataset_id VARCHAR(32),
    process_id VARCHAR(32) NULL,
    started TIMESTAMP NULL,
    finished TIMESTAMP NULL,
    message VARCHAR(256) NULL,
    PRIMARY KEY (star_id, dataset_id)
);

DROP TABLE IF EXISTS mode;
CREATE TABLE mode (
    -- mode_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY, 
    mode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frequency DOUBLE NOT NULL,
    star_id VARCHAR(32) NOT NULL,
    dataset_id VARCHAR(32) NOT NULL,
    region VARCHAR(1),
    delta_chi_squared DOUBLE,
    frequency_region_A DOUBLE,
    phase_uncertainty_jackknife DOUBLE,
    phase_uncertainty_split DOUBLE
);








