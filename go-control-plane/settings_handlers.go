package main

import "github.com/gofiber/fiber/v2"

func getSetting(c *fiber.Ctx) error {
	key := c.Params("key")
	var s AppSetting
	if err := DB.First(&s, "key = ?", key).Error; err != nil {
		return c.JSON(fiber.Map{"key": key, "value": ""})
	}
	return c.JSON(s)
}

func putSetting(c *fiber.Ctx) error {
	key := c.Params("key")
	var body struct {
		Value string `json:"value"`
	}
	if err := c.BodyParser(&body); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	s := AppSetting{Key: key, Value: body.Value}
	if err := DB.Save(&s).Error; err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(s)
}

func getSettingValue(key string) string {
	var s AppSetting
	if err := DB.First(&s, "key = ?", key).Error; err != nil {
		return ""
	}
	return s.Value
}

func putSettingValue(key, value string) {
	DB.Save(&AppSetting{Key: key, Value: value})
}
