CREATE TABLE IF NOT EXISTS `todolist` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `title` varchar(1024) NOT NULL,
  `status` int(2) NOT NULL COMMENT '是否完成',
  `create_time` int(11) NOT NULL,
  `is_training` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8;

-- insert into todolist(id, user_id, title, status, create_time) values(1, 1, '习近平五谈稳中求进织密扎牢民生保障网', '0', 1482214350), (2, 1, '特朗普获超270张选举人票将入主白 宫', '1', 1482214350);


CREATE TABLE IF NOT EXISTS `user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(24) DEFAULT NULL,
  `password` varchar(24) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4;

-- insert into user values(1, 'admin', 'admin');




CREATE TABLE IF NOT EXISTS `datalist` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `title` varchar(1024) NOT NULL,
  `status` int(2) NOT NULL COMMENT '是否完成',
  `create_time` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8;

-- insert into datalist(id, user_id, title, status, create_time) values(1, 1, '这是admin上传的一个数据', '0', 148221232), (2, 1, '特朗普获超270张选举人票将入主白 宫', '1', 1482214311);


CREATE TABLE IF NOT EXISTS `train_setting_list` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `create_time` int(11) NOT NULL,
  `model_class` int(11) NOT NULL,
  `GPU_setting` int(11) NOT NULL,
  `data_paths` varchar(1024) NOT NULL,
  `todolist_id` int(11) NOT NULL,


   `whether_data_augment` int(11) NOT NULL,
   `deep_model_class` int(11) NOT NULL,
   `ml_model_class` int(11) NOT NULL,
   `input_dim` int(11) NOT NULL,
   `output_dim` int(11) NOT NULL,
   `weight_decay` float(11) NOT NULL,
   `learning_rate` float(11) NOT NULL,
   `activation_class` int(11) NOT NULL,
   `layers_num` int(11) NOT NULL,


     `batch_size` int(11) NOT NULL,


     `optimizer` int(11) NOT NULL,
    `net_losses` int(11) NOT NULL,



  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8;


-- insert into train_setting_list(id, user_id, create_time, model_class, GPU_setting,data_paths,todolist_id) values(5, 0, 112221, 1, 148221232,'/1534863068',1535274404);
