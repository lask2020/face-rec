package main

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type SSClient struct {
	BaseURL    string
	Username   string
	Password   string
	VerifySSL  bool
	HTTPClient *http.Client
	SID        string
}

func NewSSClient(baseURL, username, password string, verifySSL bool) *SSClient {
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	if !verifySSL {
		tr := &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
		client.Transport = tr
	}

	// Remove trailing slash
	baseURL = strings.TrimSuffix(baseURL, "/")

	return &SSClient{
		BaseURL:    baseURL,
		Username:   username,
		Password:   password,
		VerifySSL:  verifySSL,
		HTTPClient: client,
	}
}

type SynoResponse struct {
	Success bool                   `json:"success"`
	Data    map[string]interface{} `json:"data"`
	Error   struct {
		Code int `json:"code"`
	} `json:"error"`
}

func (c *SSClient) Login() error {
	params := url.Values{}
	params.Add("api", "SYNO.API.Auth")
	params.Add("method", "Login")
	params.Add("version", "6")
	params.Add("account", c.Username)
	params.Add("passwd", c.Password)
	params.Add("session", "SurveillanceStation")
	params.Add("format", "sid")

	reqURL := fmt.Sprintf("%s/webapi/auth.cgi?%s", c.BaseURL, params.Encode())

	resp, err := c.HTTPClient.Get(reqURL)
	if err != nil {
		return fmt.Errorf("connection failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	var sResp SynoResponse
	if err := json.NewDecoder(resp.Body).Decode(&sResp); err != nil {
		return fmt.Errorf("decode error: %v", err)
	}

	if !sResp.Success {
		return fmt.Errorf("login failed with code %d", sResp.Error.Code)
	}

	sid, ok := sResp.Data["sid"].(string)
	if !ok || sid == "" {
		return fmt.Errorf("no SID returned")
	}

	c.SID = sid
	return nil
}

func (c *SSClient) Logout() {
	if c.SID == "" {
		return
	}
	params := url.Values{}
	params.Add("api", "SYNO.API.Auth")
	params.Add("method", "Logout")
	params.Add("version", "6")
	params.Add("session", "SurveillanceStation")
	
	reqURL := fmt.Sprintf("%s/webapi/auth.cgi?%s", c.BaseURL, params.Encode())
	c.HTTPClient.Get(reqURL)
	c.SID = ""
}

type SSCamera struct {
	ID         int    `json:"id"`
	Name       string `json:"newName"`
	OrigName   string `json:"name"`
	Model      string `json:"model"`
	Host       string `json:"host"`
	Port       int    `json:"port"`
	Status     int    `json:"status"`
	Enabled    bool   `json:"enabled"`
	Vendor     string `json:"vendor"`
	Resolution string `json:"resolution_str"`
}

func (c *SSClient) ListCameras() ([]SSCamera, error) {
	if c.SID == "" {
		return nil, fmt.Errorf("not authenticated")
	}

	params := url.Values{}
	params.Add("api", "SYNO.SurveillanceStation.Camera")
	params.Add("method", "List")
	params.Add("version", "1")
	params.Add("basic", "true")
	params.Add("streamInfo", "true")
	params.Add("privCamType", "1")
	params.Add("camStm", "1")
	params.Add("_sid", c.SID)

	reqURL := fmt.Sprintf("%s/webapi/entry.cgi?%s", c.BaseURL, params.Encode())

	resp, err := c.HTTPClient.Get(reqURL)
	if err != nil {
		return nil, fmt.Errorf("connection failed: %v", err)
	}
	defer resp.Body.Close()

	var sResp SynoResponse
	if err := json.NewDecoder(resp.Body).Decode(&sResp); err != nil {
		return nil, fmt.Errorf("decode error: %v", err)
	}

	if !sResp.Success {
		return nil, fmt.Errorf("api failed with code %d", sResp.Error.Code)
	}

	camList, ok := sResp.Data["cameras"].([]interface{})
	if !ok {
		return []SSCamera{}, nil
	}

	var cameras []SSCamera
	for _, camItem := range camList {
		camMap, ok := camItem.(map[string]interface{})
		if !ok {
			continue
		}

		idFloat, _ := camMap["id"].(float64)
		nameStr, _ := camMap["name"].(string)
		newNameStr, _ := camMap["newName"].(string)
		modelStr, _ := camMap["model"].(string)
		hostStr, _ := camMap["host"].(string)
		portFloat, _ := camMap["port"].(float64)
		statusFloat, _ := camMap["status"].(float64)
		enabledBool, _ := camMap["enabled"].(bool)
		vendorStr, _ := camMap["vendor"].(string)

		// Parse resolution
		var resStr string
		if res, exists := camMap["resolution"]; exists {
			switch v := res.(type) {
			case string:
				resStr = v
			case map[string]interface{}:
				if width, wOk := v["width"].(float64); wOk {
					if height, hOk := v["height"].(float64); hOk {
						resStr = fmt.Sprintf("%.0fx%.0f", width, height)
					}
				}
			}
		}

		c := SSCamera{
			ID:         int(idFloat),
			Name:       newNameStr,
			OrigName:   nameStr,
			Model:      modelStr,
			Host:       hostStr,
			Port:       int(portFloat),
			Status:     int(statusFloat),
			Enabled:    enabledBool,
			Vendor:     vendorStr,
			Resolution: resStr,
		}
		
		if c.Name == "" {
			c.Name = c.OrigName
		}
		if c.Name == "" {
			c.Name = "Unknown Camera"
		}

		cameras = append(cameras, c)
	}

	return cameras, nil
}
