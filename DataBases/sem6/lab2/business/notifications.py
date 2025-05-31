from repositories.redis_client import redis_client
import json
from datetime import datetime

class NotificationService:
    def __init__(self):
        self.redis = redis_client.get_connection()
        
    def send_notification(self, user_id, message, notification_type="info"):
        """
        Отправляет уведомление пользователю.
        """
        notification = {
            "type": notification_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "read": False
        }
        
        # Сохраняем уведомление в список
        self.redis.lpush(f"notifications:{user_id}", json.dumps(notification))
        # Ограничиваем список 100 последними уведомлениями
        self.redis.ltrim(f"notifications:{user_id}", 0, 99)
        
        # Публикуем событие в канал
        self.redis.publish(f"user:{user_id}:notifications", json.dumps(notification))

    def get_unread_notifications(self, user_id, mark_as_read=False):
        """
        Получает непрочитанные уведомления пользователя.
        """
        notifications = []
        all_notifications = self.redis.lrange(f"notifications:{user_id}", 0, -1)
        
        for notification_json in all_notifications:
            notification = json.loads(notification_json)
            if not notification.get("read", True):
                notifications.append(notification)
                
                if mark_as_read:
                    notification["read"] = True
                    # Обновляем уведомление в списке
                    self.redis.lset(f"notifications:{user_id}", 
                                  all_notifications.index(notification_json), 
                                  json.dumps(notification))
        
        return notifications

    def mark_all_as_read(self, user_id):
        """
        Помечает все уведомления пользователя как прочитанные.
        """
        all_notifications = self.redis.lrange(f"notifications:{user_id}", 0, -1)
        
        for notification_json in all_notifications:
            notification = json.loads(notification_json)
            if not notification.get("read", True):
                notification["read"] = True
                # Обновляем уведомление в списке
                self.redis.lset(f"notifications:{user_id}", 
                              all_notifications.index(notification_json), 
                              json.dumps(notification))
                
    def get_all_notifications(self, user_id):
        """
        Получает все уведомления пользователя (и прочитанные, и непрочитанные).
        """
        notifications = []
        all_notifications = self.redis.lrange(f"notifications:{user_id}", 0, -1)
        
        for notification_json in reversed(all_notifications):  # Новые сверху
            try:
                notifications.append(json.loads(notification_json))
            except:
                continue
                
        return notifications
    
    def mark_as_read(self, user_id, timestamp):
        """
        Помечает конкретное уведомление как прочитанное по timestamp.
        """
        all_notifications = self.redis.lrange(f"notifications:{user_id}", 0, -1)
        
        for notification_json in all_notifications:
            notification = json.loads(notification_json)
            if notification.get("timestamp") == timestamp:
                notification["read"] = True
                # Обновляем уведомление в списке
                self.redis.lset(f"notifications:{user_id}", 
                              all_notifications.index(notification_json), 
                              json.dumps(notification))
                break