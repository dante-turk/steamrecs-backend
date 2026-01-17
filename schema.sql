DELETE FROM user_games
WHERE playtime_forever_minutes < 300;

DELETE FROM games
WHERE appid NOT IN (SELECT DISTINCT appid FROM user_games);

VACUUM;
