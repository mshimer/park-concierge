-- Park-pass holders and their visit history: the "internal system of record" the agent queries
-- through MCP. Read-only from the agent's point of view.

CREATE TABLE customer (
    id                  SERIAL PRIMARY KEY,
    name                TEXT NOT NULL,
    email               TEXT UNIQUE NOT NULL,
    pass_tier           TEXT NOT NULL CHECK (pass_tier IN ('none','annual','senior','access')),
    home_state          CHAR(2) NOT NULL,
    party_size          INT NOT NULL DEFAULT 2,
    accessibility_needs TEXT,              -- free text; NULL means none stated
    max_elevation_ft    INT,               -- NULL means no limit; the evals check this is honoured
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE visit (
    id          SERIAL PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES customer(id) ON DELETE CASCADE,
    park_code   TEXT NOT NULL,             -- NPS code; joins to Chroma metadata.park_code
    visited_on  DATE NOT NULL,
    rating      INT CHECK (rating BETWEEN 1 AND 5)
);
CREATE INDEX ON visit (customer_id);

-- Explicit ids so eval prompts can name a customer by number.
INSERT INTO customer (id, name, email, pass_tier, home_state, party_size, accessibility_needs, max_elevation_ft) VALUES
  (1042, 'Maria Delgado',   'maria.delgado@example.com',  'access', 'TX', 3, 'wheelchair-accessible trails and paved overlooks only; cannot manage stairs or ladders', 5000),
  (1043, 'Tom Ishikawa',    'tom.ishikawa@example.com',   'annual', 'WA', 2, NULL, NULL),
  (1044, 'Priya Raman',     'priya.raman@example.com',    'annual', 'CA', 4, NULL, NULL),
  (1045, 'Walter Boone',    'walter.boone@example.com',   'senior', 'FL', 2, 'limited walking distance, under half a mile', 4000),
  (1046, 'Noor Haddad',     'noor.haddad@example.com',    'annual', 'NY', 1, NULL, NULL),
  (1047, 'Greg Lindqvist',  'greg.lindqvist@example.com', 'none',   'MN', 5, NULL, 9000),
  (1048, 'Aisha Bello',     'aisha.bello@example.com',    'annual', 'IL', 2, NULL, NULL),
  (1049, 'Dmitri Volkov',   'dmitri.volkov@example.com',  'annual', 'CO', 2, NULL, NULL),
  (1050, 'Hannah Cross',    'hannah.cross@example.com',   'senior', 'AZ', 2, 'service animal travelling with party', 7000),
  (1051, 'Luis Ferreira',   'luis.ferreira@example.com',  'none',   'NV', 3, NULL, NULL),
  (1052, 'Beth Okonkwo',    'beth.okonkwo@example.com',   'annual', 'OR', 2, NULL, NULL),
  (1053, 'Sam Whitfield',   'sam.whitfield@example.com',  'access', 'ME', 4, 'hearing impaired; needs captioned or written program materials', NULL);

INSERT INTO visit (customer_id, park_code, visited_on, rating) VALUES
  (1042, 'whsa', '2024-03-14', 5),
  (1042, 'bibe', '2023-11-02', 4),
  (1042, 'sagu', '2023-02-19', 5),
  (1043, 'mora', '2024-08-09', 5),
  (1043, 'noca', '2024-07-21', 4),
  (1043, 'olym', '2023-09-30', 5),
  (1044, 'yose', '2024-06-11', 5),
  (1044, 'sequ', '2024-06-14', 4),
  (1044, 'jotr', '2023-12-27', 4),
  (1045, 'ever', '2024-01-23', 4),
  (1045, 'bisc', '2023-12-05', 3),
  (1045, 'drto', '2023-03-18', 5),
  (1046, 'acad', '2024-10-07', 5),
  (1046, 'shen', '2024-05-16', 4),
  (1047, 'voya', '2024-07-04', 5),
  (1047, 'isro', '2023-08-12', 5),
  (1047, 'grsm', '2022-10-19', 4),
  (1048, 'indu', '2024-06-29', 3),
  (1048, 'cuva', '2024-04-13', 4),
  (1049, 'romo', '2024-09-02', 5),
  (1049, 'blca', '2024-05-25', 4),
  (1049, 'grsa', '2023-06-17', 4),
  (1049, 'meve', '2022-09-10', 5),
  (1050, 'grca', '2024-04-05', 5),
  (1050, 'pefo', '2024-03-22', 3),
  (1050, 'sagu', '2023-01-30', 4),
  (1051, 'deva', '2024-02-17', 5),
  (1051, 'grba', '2023-07-08', 4),
  (1052, 'crla', '2024-08-24', 5),
  (1052, 'redw', '2024-05-11', 5),
  (1053, 'acad', '2024-09-15', 4),
  (1053, 'cuva', '2023-07-26', 3);

-- GOTCHA: explicit-id inserts do not advance a SERIAL's sequence, so the next plain INSERT would
-- collide on id 1. Fix it here rather than leaving a landmine for anyone who adds a row later.
SELECT setval(pg_get_serial_sequence('customer','id'), (SELECT max(id) FROM customer));
SELECT setval(pg_get_serial_sequence('visit','id'),    (SELECT max(id) FROM visit));
